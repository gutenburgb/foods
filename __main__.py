# pip install spoonacular
import spoonacular as sp
from pprint import pprint
import pymongo
from urllib.parse import quote_plus
import arrow
from apscheduler.schedulers.blocking import BlockingScheduler  # Scheduler  # conda install -c conda-forge apscheduler
import json

db_url = 'ec2-3-86-197-221.compute-1.amazonaws.com'
db_name = 'foods'
db_user = 'myScraper'
db_pass = 'irj482dFDWQ2**ddc'

coll_name = 'recipes_small'
coll_track_name = 'cuisine_track_small'
coll_api_name = 'api_track_sans_small'

points_remaining = 150  # free API keys get this many points (this is used to initialize the database)

api_keys = [
    '77f3addc46df44779f401f6702156762', #L
    '6b23d8d7b7e64a4db3a51d1296712a87', #K
    '6bebedc7fa924bd1997078abfdf1c90f'  #J
    # A
    # S
]

all_cuisines = ['African', 'American', 'British', 'Cajun', 'Caribbean', 'Chinese', 'Eastern European', 'European',
                'French', 'German', 'Greek', 'Indian', 'Irish', 'Italian', 'Japanese', 'Jewish', 'Korean',
                'Latin American', 'Mediterranean', 'Mexican', 'Middle Eastern', 'Nordic', 'Southern', 'Spanish',
                'Thai', 'Vietnamese']

points_per_query = 4.5
# single API call to 'search' endpoint? 1 point
# fillIngredients == true? then + 0.025 points per recipe
# addRecipeInformation == true? then + 0.025 points per recipe
# addRecipeNutrition == true? then + 0.025 points per recipe
# + 0.01 points per result returned (100)

# one call for 100 recipes with 3 options = 100 * (0.025 + 0.025 + 0.025 + 0.01) + 1 = 9.5 points
# daily API points = 150
# daily queries per API at this rate = 15.7 (or 15) returning 1500 recipes (perhaps 70 more, if we squeeze)

# smaller (small) (sans addRecipeInformation and addRecipeNutrition
# one call for 100 recipes with one option = 100 * (0.025 + 0.01) + 1 = 4.5 points

def initialize_api_defaults(coll_api):
    for key in api_keys:
        if not coll_api.find_one({'key': key}):
            coll_api.insert_one({
                'key': key,
                'points_remaining': points_remaining,
                'date_last_queried': arrow.utcnow().shift(hours=-25).datetime
            })

def initialize_tracking_defaults(coll_track):
    # only bothers the db if records are missing
    for cuisine in all_cuisines:
        if not coll_track.find_one({'cuisine': cuisine}):
            coll_track.insert_one({
                'cuisine': cuisine,
                'count': 0,
                'has_more': True,
            })


def adjust_point_values(coll_api):
    # for all api keys in coll_track
    # if date_last_queried is > 24 hours ago, then reset points_remaining
    results = coll_api.find({})
    for result in results:
        db_date = arrow.Arrow.fromdatetime( result.get('date_last_queried', None) ) # make tzaware
        db_key = result.get('key', 'x')
        now = arrow.utcnow()
        time_diff = (now - db_date)
        seconds = time_diff.days * 86400 + time_diff.seconds  # 86400 = seconds in a day
        if seconds > 86401:  # 86400 = seconds in a day
            coll_api.update_one(
                {'key': db_key},  # filter: find this one
                {
                    '$set': {
                        'points_remaining': points_remaining
                    }
                }
            )


def key_has_points_remaining(key, coll_api):
    return coll_api.find_one({'key': key}, {'points_remaining': 1}).get('points_remaining', -1) > points_per_query


class Cuisine:
    offset_size = 100

    data = {}  # from mongodb

    def __init__(self, api, coll, coll_track, coll_api, name):
        # poor man's ORM
        self.api = api
        self.coll = coll
        self.coll_track = coll_track
        self.coll_api = coll_api
        self.name = name
        self.data = self.__from_mongo(name)

    def __from_mongo(self, cuisine_name):
        results = self.coll_track.find_one({'cuisine': cuisine_name})
        if results:
            data = results
        else:
            data = {
                'cuisine': cuisine_name,
                'count': 0,
                'has_more': True
            }
        return data

    def query(self):
        # refresh from mongodb
        self.data = self.__from_mongo(self.name)

        if self.data.get('has_more', False):
            if key_has_points_remaining(self.api.api_key, self.coll_api):
                calculated_offset = self.data.get('count', 0)

                response_raw = self.api.search_recipes_complex(
                    query=".*",
                    cuisine='African',  #self.data.get('cuisine', 'American'),
                    #addRecipeInformation=True,
                    #addRecipeNutrition=True,
                    fillIngredients=True,
                    sort='popularity',
                    offset=calculated_offset,
                    number=self.offset_size
                )
                if response_raw.status_code != 200:
                    print(f"ERROR: non-200 status code returned from query: {response_raw.status_code}")

                response_full = json.loads(response_raw.content.decode('utf-8'))
                response = response_full.get('results', [])

                # identify if all 'n' results were returned, if not, change 'has_more' to false
                if len(response) < self.offset_size:
                    self.data['has_more'] = False

                # update mongodb with data found
                if response:
                    insert_result = self.coll.insert_many(response)
                    print(f'Inserted {len(insert_result.inserted_ids)} objects...')

                    # update mongodb tracking collection with new info
                    self.coll_track.update_one(
                        {'cuisine': self.name},  # filter: find this one
                        {
                            '$set': {
                                 'has_more': self.data.get('has_more', False)
                             },
                            '$inc': {
                                'count': len(insert_result.inserted_ids)  # decrement the count of recipes
                            }
                        }
                    )

                    # update mongodb api tracking with new info
                    self.coll_api.update_one(
                        {'key': self.api.api_key},
                        {
                            '$set': {
                                'date_last_queried': arrow.utcnow().datetime
                            },
                            '$inc': {
                                'points_remaining': -1 * points_per_query  # decrement the remaining points
                            }
                        }
                    )
                else:
                    print(f"No response received from query on cuisine type: {self.data.get('cuisine')}")
            else:
                print(f"Not enough points left ({self.data.get('points_remaining', -1)}) " +
                      f"on this API key ({self.api.api_key}) to query...")
        else:
            print(f"has_more was False. Refusing to query. " +
                  f"It is unlikely that more recipes exist in this cuisine category: {self.data.get('cuisine')}")


def main():
    # defaults to port 27017
    client = pymongo.MongoClient(f'mongodb://{db_user}:{quote_plus(db_pass)}@{db_url}/{db_name}')

    db = client.foods
    print(f'Available collections: {db.list_collection_names()}')
    print(f'Using collection: {coll_name}')
    coll = db.get_collection(coll_name)
    print(f'There are {coll.count_documents({})} documents in this collection.')

    coll_track = db.get_collection(coll_track_name)
    coll_api = db.get_collection(coll_api_name)

    pprint(list(coll_track.find({'i': {'$gt': -1}})))  # prints all cuisine tracking info

    initialize_tracking_defaults(coll_track)

    initialize_api_defaults(coll_api)

    # create list of cuisine objects
    cuisine_objs = {}  # by cuisine name
    for cuisine_name in all_cuisines:
        per_api = []
        for key in api_keys:
            api = sp.API(key)
            per_api.append( Cuisine(api, coll, coll_track, coll_api, cuisine_name) )
        cuisine_objs[cuisine_name] = per_api

    # cuisine_objs:
    # {
    #   'American': [ Cuisine(API1, American), Cuisine(API2, American), Cuisine(API3, American) ]
    #   'Chinese': [ Cuisine(API1, Chinese), Cuisine(API2, Chinese), Cuisine(API3, Chinese) ]
    # }

    # task:
    # pick a cuisine with least amount of recipes, but with has_more == True
    # run query for that cuisine
    sched = BlockingScheduler()

    # use apscheduler to schedule a task every 1 minute
    @sched.scheduled_job('interval', seconds=60)
    def task():
        results = list( coll_track.find({'has_more': True}).sort([('count', pymongo.ASCENDING)]) )
        if len(results) > 0:
            first_result = results[0]
            chosen_cuisine = first_result.get('cuisine', '')
            if chosen_cuisine:
                # get an API that has available bandwidth
                apis = cuisine_objs.get(chosen_cuisine, None)
                if apis:
                    chosen_api = None
                    for api in apis:
                        if key_has_points_remaining(api.api.api_key, coll_api):
                            chosen_api = api
                            break
                    if chosen_api is None:
                        print(f"WARN: No API found with adequate points remaining")
                    else:
                        # this API can now be told to query
                        chosen_api.query()
                else:
                    print(f"Unable to retrieve any APIs for cuisine {chosen_cuisine}")
            else:
                print(f"Error retrieving cuisine tracking info in this iteration.")
        else:
            print(f"No viable cuisines found to query for")
        adjust_point_values(coll_api)  # handles resetting API key point tracking

    # run the task immediately
    task()

    # start scheduler
    sched.start()  # blocking operation


if __name__ == '__main__':
    main()
