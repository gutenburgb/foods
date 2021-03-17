from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pathlib import Path
from sklearn.feature_extraction import text  # for adding additional stopwords
import textwrap  # for truncating reviews down to a similar size
import pandas as pd
import numpy as np
import re

pd.options.plotting.backend = "plotly"

from nltk.stem.porter import PorterStemmer
STEMMER=PorterStemmer()

more_stopwords = text.ENGLISH_STOP_WORDS.union({
    'recipe', 'w', 'use', 'thankful', 'recioe', 'great', 'majorly'
})

# bin by recipe

def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(words) for words in words]
    return words

from nltk.stem import WordNetLemmatizer
LEMMER = WordNetLemmatizer()

def MY_LEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [LEMMER.lemmatize(words) for words in words]
    return words


def fit_lda(df, num_topics=5):
    # takes a vectorized df (just the vectors)
    lda = LatentDirichletAllocation(n_components=num_topics,
                                    max_iter=50,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0,
                                    batch_size=128)
    lda_z = lda.fit_transform(df)
    return lda, lda_z

def analyze_lda(df_display, lda, lda_z):
    # lda_z (5000 rows)
    # words -> (9000 or so) lda.components_
    winning_categories = np.array(lda_z).argmax(axis=1)
    print('Category distributions: \n', pd.DataFrame(winning_categories).value_counts() )

    pass

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        word_tuples = [(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]]
        word_tuples = [(x[0], round(x[1],3)) for x in word_tuples]
        print(word_tuples)

def vectorize_content(df, text_col='text'):
    global more_stopwords  # forgive me my sins
    my_preproc = MY_LEMMER #  set to None or MY_STEMMER

    if my_preproc:
        # we must also pre-process the stopword list
        more_stopwords = set(my_preproc(' '.join(more_stopwords)))
    # get frequencies using CountVectorizer
    cv = CountVectorizer(
        input='content',
        stop_words=more_stopwords,
        min_df=3,
        tokenizer=MY_LEMMER,
        lowercase=True,)

    # cv = CountVectorizer(
    #     input='content',
    #     stop_words='english',
    #     tokenizer=MY_LEMMER,
    #     lowercase=True,)

    # cv = CountVectorizer(
    #     input='content',
    #     stop_words='english',
    #     token_pattern=r"(?u)\b\w\w+\b") # preserves symbols: r"(?u)\b\w\w+\b|!|\?|\"|\'"

    # cv = TfidfVectorizer(  input="content",
    #                        analyzer='word',
    #                        stop_words=more_stopwords,
    #                        tokenizer=MY_LEMMER,
    #                        lowercase=True,
    #                        use_idf=False
    #                        )

    w = cv.fit_transform(list(df[text_col]))

    # convert to DataFrame without label
    df_vectors = pd.DataFrame(w.A, columns=cv.get_feature_names())

    # uppercase columns that aren't from the df_vectors dataframe (avoids naming conflicts)
    df.columns = map(str.upper, df.columns)

    # add labels (combine both data frames)
    df = df.reset_index()
    df = pd.concat([df.loc[:, df.columns != text_col.upper()], df_vectors], axis=1)
    return df, cv

def main(make_graphs=False):
    # run with defaults from my system
    path_project = Path.cwd()
    path_data = path_project / 'data'
    path_reviews = path_data / 'clean_reviews.csv'
    df_reviews_all = pd.read_csv(path_reviews, index_col=0)

    #TODO: remove &quot;, &QUOT;, html &#999;, \r\n, and escaped quotes

    # df_reviews_all.groupby('recipe_id').filter(lambda x: len(x) > 5)
    reviews_per_recipe = df_reviews_all.groupby('recipe_id').size().nlargest(1000000000000)

    if make_graphs:
        reviews_per_recipe.hist().show()

    # let's look at recipes that have 50 reviews
    chosen_recipes = reviews_per_recipe[reviews_per_recipe == 50]
    df_reviews = df_reviews_all[df_reviews_all['recipe_id'].isin(chosen_recipes.index)]

    # truncate to a more usable amount!
    #df_reviews = df_reviews_all.sample(5000, random_state=123)


    print(f'There are now {len(df_reviews)} reviews.')

    # convert to '' or drop nans
    dropna = False
    text_col = 'review'
    nan_count = df_reviews[text_col].isna().sum()
    if nan_count > 0:
        if dropna:
            print(f'WARNING: Dropping {nan_count} values from {text_col} column because: nan.')
            df_reviews = df_reviews.dropna(subset=[text_col])
        else:
            print(f"WARNING: setting {nan_count} values to '' from {text_col} column because: nan.")
            df_reviews[df_reviews[text_col].isna()] = ''

    # feature engineering: length of review by character
    df_reviews['len'] = df_reviews['review'].str.len()

    # boxplot of lengths of each review
    if make_graphs:
        df_reviews.boxplot(column=['len'])

    # vars
    char_review_length_thresh = 50

    # remove any reviews shorter than the threshold
    i_reviews_too_short = len(df_reviews[df_reviews['len'] < char_review_length_thresh])
    df_reviews = df_reviews[df_reviews['len'] >= char_review_length_thresh]

    # remove top x% of reviews by length, truncating them down to something shorter
    q = df_reviews['len'].quantile(0.90)  # 0.75 = 75%
    i_reviews_too_long = len(df_reviews[df_reviews['len'] < q])
    #df_reviews['review'] = df_reviews[df_reviews['len'] < q]['review']
    df_reviews['review'] = df_reviews['review'].apply(lambda s: textwrap.shorten(s, width=q, placeholder=''))

    print(f'Truncating {i_reviews_too_long} long reviews and removing {i_reviews_too_short} short reviews')
    print(f'There are now {len(df_reviews)} reviews.')
    pass

    lda_for_single_recipes(df_reviews)
    # lda_for_ratings(df_reviews)


    pass

def lda_for_single_recipes(df_reviews):
    df_reviews_v, cv = vectorize_content(df_reviews, 'review')
    df_reviews_v = df_reviews_v.drop(columns='index')

    meta = ['RECIPE_ID', 'RATING', 'LEN']  # don't send these columns to LDA algorithm
    group = df_reviews_v.groupby('RECIPE_ID')
    group.get_group((list(group.groups)[0]))
    for i in range(len(group.groups)):
        recipe_id = list(group.groups)[i]
        print(f"### group: {i}, recipe_id: {recipe_id} ###")
        df_tmp = df_reviews_v[df_reviews_v['RECIPE_ID'] == recipe_id]
        print(f'Number of reviews in group {i}: {len(df_tmp)}')
        lda, lda_z = fit_lda(df_tmp.drop(columns=meta), num_topics=5)
        analyze_lda(df_reviews, lda, lda_z)
        print_topics(lda, cv, 7)
        print('')


def lda_for_ratings(df_reviews):
    df_reviews_v, cv = vectorize_content(df_reviews, 'review')
    df_reviews_v = df_reviews_v.drop(columns='index')

    meta = ['RECIPE_ID', 'RATING', 'LEN']  # don't send these columns to LDA algorithm
    for i in range(6):
        print(f"### {i} ###")
        df_tmp = df_reviews_v[df_reviews_v['RATING'] == i]
        print(f'Number of reviews with rating {i}: {len(df_tmp)}')
        lda, lda_z = fit_lda(df_tmp.drop(columns=meta), num_topics=5)
        analyze_lda(df_reviews, lda, lda_z)
        print_topics(lda, cv, 7)
        print('')

if __name__ == '__main__':
    main()