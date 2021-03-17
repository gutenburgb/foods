### Queries MongoDB to retrieve recipes ###

# Would recommend using a Conda environment (Anaconda)
# Use python 3.8.5 if you'd like
# conda create -n foods python=3.8.5
# conda activate foods
# conda install pymongo urllib3 arrow
#
# python query.py

import pymongo
from urllib.parse import quote_plus
from pprint import pprint
import pandas as pd

db_url = 'ec2-3-86-197-221.compute-1.amazonaws.com'
db_name = 'foods'
db_user = 'myReader'
db_pass = 'woi3jT20**ahako22'
coll_name = 'recipes'

all_cuisines = ['African', 'American', 'British', 'Cajun', 'Caribbean', 'Chinese', 'Eastern European', 'European',
                'French', 'German', 'Greek', 'Indian', 'Irish', 'Italian', 'Japanese', 'Jewish', 'Korean',
                'Latin American', 'Mediterranean', 'Mexican', 'Middle Eastern', 'Nordic', 'Southern', 'Spanish',
                'Thai', 'Vietnamese']


def main():
    # defaults to port 27017
    client = pymongo.MongoClient(f'mongodb://{db_user}:{quote_plus(db_pass)}@{db_url}/{db_name}')

    db = client.foods
    print(f'Available collections: {db.list_collection_names()}')
    print(f'Using collection: {coll_name}')
    coll = db.get_collection(coll_name)
    print(f'There are {coll.count_documents({})} documents in this collection.')

    # Query MongoDB for all African recipes
    african_recipes_result = coll.find({'cuisines': ['African']})

    # Query for ALL recipes
    # african_recipes_result = coll.find({})

    # ACTUALLY retrieve these recipes. Note that if you don't need all info, you should iterate over
    # the generator: 'african_recipes_result' as this will retrieve one result at a time as you use it,
    # so you won't have to deal with a TON of data all at once. list(...) resolves all at once, and can take time
    african_recipes = list(african_recipes_result)
    print(f"{len(african_recipes)} recipes found.")

    # display object in a prettier format (pprint)
    # if len(african_recipes) > 0:
    #     pprint(african_recipes[0])

    recipes = african_recipes

    # maps from spoonacular column name -> our preferred name.
    col_map = {
        '_id': 'recipe_id',
        'cuisines': 'cuisine_type',
        'id': 'ingredient_id',
        'name': 'ingredient_name',
        'nameClean': 'ingredient_name_clean',
        'originalName': 'ingredient_name_orig',
        'unit': 'unit',
        'amount': 'amount'
        }

    def sp_transform(recipes, col_map):
        from_cols = list(col_map.keys())
        for recipe in recipes:
            d = {}  # create a dict for yielding
            for ingr in recipe.get('extendedIngredients', []):
                # handle some rows specially (some are strings, some are arrays)
                for col in from_cols:
                    if col == '_id':
                        d[col] = str(recipe.get('_id', 'x'))
                    if col == 'cuisines':
                        d[col] = recipe.get('cuisines', [])
                    if col in ingr.keys():
                        d[col] = ingr[col]  # grab-bag for all other columns
                # make one row per ingredient
                yield {col_map.get(x, x): y for x, y in d.items()}  # convert names using col_map

    # convert to dataframe. Create a row for each INGREDIENT in the recipe
    df = pd.DataFrame(data=sp_transform(recipes, col_map), columns=list(col_map.values()))

    # note that recipes can have multiple cuisine types and some of them seem to overlap for example
    # ['Mediterranean', 'European', 'Greek']
    # ['German', 'European']
    df.head()


if __name__ == '__main__':
    main()
