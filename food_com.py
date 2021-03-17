# WHICH Food.com? THIS Food.com: https://www.kaggle.com/irkaal/foodcom-recipes-and-reviews
from pathlib import Path
from pprint import pprint
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

import gensim
from gensim import corpora
from pprint import pprint


# conda install -c conda-forge scikit-learn
# conda install pyarrow fastparquet   # required for importing parquet files
# conda install gensim   # topic modeling


def load_data(path_reviews, path_recipes):
    path_reviews = Path(path_reviews)
    path_recipes = Path(path_recipes)

    # ['ReviewId', 'RecipeId', 'AuthorId', 'AuthorName', 'Rating', 'Review', 'DateSubmitted', 'DateModified']
    df_reviews_all = pd.read_parquet(path_reviews)  # 1,401,982 rows, 8 columns

    # ['RecipeId', 'Name', 'AuthorId', 'AuthorName', 'CookTime', 'PrepTime',
    #        'TotalTime', 'DatePublished', 'Description', 'Images', 'RecipeCategory',
    #        'Keywords', 'RecipeIngredientQuantities', 'RecipeIngredientParts',
    #        'AggregatedRating', 'ReviewCount', 'Calories', 'FatContent',
    #        'SaturatedFatContent', 'CholesterolContent', 'SodiumContent',
    #        'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent',
    #        'RecipeServings', 'RecipeYield', 'RecipeInstructions']
    df_recipes_all = pd.read_parquet(path_recipes)  # 522,517 rows, 28 columns

    # truncate to a more usable amount!
    df_reviews = df_reviews_all.sample(5000, random_state=123)
    df_recipes = df_recipes_all.sample(5000, random_state=123)

    # df_reviews
    # ReviewId, RecipeId, Rating, Review (for sentiment and LDA), DateSubmitted (this is all columns except AuthorId, AuthorName and DateModified
    l_reviews_cols = ['ReviewId', 'RecipeId', 'Rating', 'Review', 'DateSubmitted']

    # df_recipes
    # Name, RecipeId, AggregatedRating, ReviewCount, RecipeIngredientParts, DatePublished
    # RecipeIngredientParts is an already tokenized list of strings, but could be stemmed or lemmatized. Needs whitespace trimming
    # do we want RecipeIngredientQuantities? or Keywords? RecipeCategory is one of "Asian, Dessert, Whole Chicken, Vegetable
    l_recipes_cols = ['Name', 'RecipeId', 'AggregatedRating', 'ReviewCount', 'RecipeIngredientParts', 'DatePublished']

    # trim down to desired columns
    df_reviews = df_reviews[l_reviews_cols]
    df_recipes = df_recipes[l_recipes_cols]

    # throw out Rating=0 because a lot of those seem to be mistakes (0 rating, but praised for how good it tastes)
    print(f"Removed {len(df_reviews[df_reviews['Rating'] == 0])} of {len(df_reviews)} rows because Rating = 0")
    df_reviews = df_reviews[df_reviews['Rating'] != 0]

    # combine the two datasets into one row per recipe+review combo
    df = df_recipes.join(df_reviews, on='RecipeId', rsuffix='r_', how='left')

    # This shouldn't be necessary if *all* data is joined
    # WARNING: REMOVES ALL BLANK REVIEWS

    df = df.dropna(subset=['Review'])



    # text = ["I like to play Football",
    #         "Football is the best game",
    #         "Which game do you like to play ?"]

    # tokens = [sentence.split() for sentence in df['Review']]
    tokens = [[token for token in sentence.split()] for sentence in df['Review']]

    gensim_dictionary = corpora.Dictionary()
    gensim_corpus = [gensim_dictionary.doc2bow(token, allow_update=True) for token in tokens]

    from gensim import models
    import numpy as np

    tfidf = models.TfidfModel(gensim_corpus, smartirs='ntc')

    for sent in tfidf[gensim_corpus]:
        print([[gensim_dictionary[id], np.around(frequency, decimals=2)] for id, frequency in sent])


    return df

def main():
    path_data_dir = Path(Path().cwd() / 'data')
    path_reviews = Path(path_data_dir / 'reviews.parquet')
    path_recipes = Path(path_data_dir / 'recipes.parquet')
    load_data(path_reviews, path_recipes)

if __name__ == '__main__':
    main()
