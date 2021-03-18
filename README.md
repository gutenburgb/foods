some results

### 5 (cv with lemma) ###
Number of reviews with rating 5: 3461
Category distributions: 
 3    3415
4      32
1       5
2       5
0       4
dtype: int64
Topic 0:
[('spice', 9.823871869486728), ('amp', 8.342459374669028), ('tentativeness', 6.071832104734304), ('seasings', 5.453452026456891), ('parchment', 4.97378656941154), ('swerve', 4.269958814039458), ('amy', 4.195514942376956), ('mkt', 4.179213066813273), ('morton', 4.0699727939846015), ('til', 3.8010889301422854)]
Topic 1:
[('wine-', 9.435811349115024), ('class', 9.002676101484585), ('jig-saw', 8.69708065697364), ('dean', 7.80298854247425), ('minature', 6.241477128440465), ('smuggle', 5.309367235860495), ('candied', 5.054824067159628), ('slipthem', 5.0023058981434625), ('eastern', 4.9971571717244), ('nummy', 4.963412703812208)]
Topic 2:
[('uncharacteristically', 10.737121636556154), ('visible', 7.82828677129154), ('oklahoma', 5.983352021427208), ('raise', 4.085573118554567), ('thank', 3.8668405552028515), ('pr', 3.184816242251664), ('drizzle', 3.149365461281274), ('handle', 3.116241720124049), ('darn', 3.075623307389867), ('skirt', 3.048971748248316)]
Topic 3:
[('w', 2394.3622074490904), ('recioe', 2284.195269904964), ('use', 1417.6576195158664), ('thankful', 1220.640108956465), ('majorly', 1147.3258805206162), ('great', 1102.233217249072), ('szechuan', 1069.6918990771098), ('easy', 760.9716764391087), ('rz', 732.6743320548641), ('good', 697.4478429136856)]
Topic 4:
[('chocolate', 155.22488626158645), ('cup', 121.00279096747207), ('mileage', 86.6135366365838), ('flour', 77.1529983538593), ('cream', 63.60915218505958), ('chip', 57.452064124175905), ('whatcha', 35.123941185914774), ('vanila', 32.983245818529355), ('whisky', 27.13155960658869), ('bar', 24.424411925309776)]



### 5 tfidf use_idf=True ###
Number of reviews with rating 5: 3461
Category distributions: 
 4    3454
2       3
3       2
0       1
1       1
dtype: int64
Topic 0:
[('-orchid', 0.6113056352833837), ('elmo', 0.6113056293524919), ('skiing', 0.5066338589244849), ('disecting', 0.5066338588845337), ('confident', 0.506633858189508), ('reeses', 0.5066338536838408), ('tartness', 0.4923556651712044)]
Topic 1:
[('sidedish', 0.8500747932478389), ('easy-', 0.8500747924529811), ('rosemary', 0.8030907972655671), ('morty', 0.7163100323584564), ('barbque', 0.6748738492775412), ('wear', 0.5483024118209749), ('glove', 0.5483021853672301)]
Topic 2:
[('skirt', 0.9505692514990817), ('work', 0.798445457880168), ('elevates', 0.7501980170210075), ('row', 0.7421977699251127), ('winners-pillsbury', 0.7281258444062751), ('traceymae', 0.7246786781542104), ('-p', 0.698118618412999)]
Topic 3:
[('sight', 1.0259973501117106), ('pom', 0.7278779603794212), ('sediment', 0.722393369486026), ('loco', 0.7223933690601061), ('lumpia', 0.7098067565634596), ('az', 0.7082000404151574), ('awfully', 0.7002359724901871)]
Topic 4:
[('w', 165.54682958747466), ('recioe', 164.4473208170775), ('thankful', 120.66440635036157), ('great', 117.08449896748795), ('use', 113.05458242583688), ('majorly', 109.55645495996951), ('easy', 93.59826206082633)]





### 5 tfidf use_idf=False ###
Number of reviews with rating 5: 3461
Category distributions: 
 4    3458
3       2
2       1
dtype: int64
Topic 0:
[('yummmmmmmmmmmmmmmmmmmmmmmmmmy', 0.2), ('e-mailed', 0.2), ('unit', 0.2), ('pizza', 0.2), ('yummmmmmmmm', 0.2), ('ethnic', 0.2), ('dew', 0.2)]
Topic 1:
[('yummmmmmmmmmmmmmmmmmmmmmmmmmy', 0.2), ('unit', 0.2), ('e-mailed', 0.2), ('pizza', 0.2), ('yummmmmmmmm', 0.2), ('ethnic', 0.2), ('dew', 0.2)]
Topic 2:
[('-p', 0.614), ('shame', 0.614), ('hamwow-ed', 0.614), ('everyday', 0.213), ('holiday', 0.203), ('vco', 0.201), ('yummmmmmmmmmmmmmmmmmmmmmmmmmy', 0.2)]
Topic 3:
[('az', 0.648), ('sediment', 0.64), ('loco', 0.64), ('exact', 0.605), ('vampire', 0.599), ('sight', 0.58), ('canada', 0.557)]
Topic 4:
[('w', 426.18), ('recioe', 421.696), ('thankful', 251.393), ('use', 246.567), ('majorly', 222.264), ('great', 217.877), ('szechuan', 181.028)]





### 5 use_idf=False + additional stopwords ###
Number of reviews with rating 5: 3461
Category distributions: 
 3    3457
4       2
1       1
2       1
dtype: int64
Topic 0:
[('yummmmmmmmmmmmmmmmmmmmmmmmmmy', 0.2), ('e-mailed', 0.2), ('unit', 0.2), ('easy-', 0.2), ('sidedish', 0.2), ('pizza', 0.2), ('yummmmmmmmm', 0.2)]
Topic 1:
[('vampire', 0.599), ('sight', 0.579), ('wow', 0.563), ('yorkshire', 0.563), ('two-layer', 0.555), ('canada', 0.554), ('forgot', 0.205)]
Topic 2:
[('coverage', 0.495), ('youngster', 0.495), ('dye', 0.494), ('fountain', 0.49), ('health', 0.48), ('hair', 0.247), ('eye', 0.239)]
Topic 3:
[('vowed', 438.222), ('thankd', 261.822), ('usa', 253.563), ('majoram', 229.187), ('szechuan', 185.445), ('easy', 164.599), ('good', 146.346)]
Topic 4:
[('hee', 0.817), ('az', 0.704), ('exact', 0.696), ('loco', 0.693), ('sediment', 0.693), ('scarce', 0.512), ('solved', 0.508)]





### 5 use_idf=False + additional stopwords + only recipes with 50 reviews (this is a whole sample tho) ###
Number of reviews with rating 5: 2382
Category distributions: 
 3    2382
dtype: int64
Topic 0:
[('shouldn', 0.5), ('executed', 0.5), ('clever', 0.473), ('household', 0.206), ('raf', 0.204), ('stank', 0.202), ('simmering', 0.2)]
Topic 1:
[('addicted', 0.2), ('mop', 0.2), ('carve', 0.2), ('wasting', 0.2), ('bothe', 0.2), ('zucchini', 0.2), ('permanent', 0.2)]
Topic 2:
[('ut', 0.453), ('plated', 0.453), ('ingridents', 0.453), ('unimpressed', 0.453), ('compilation', 0.453), ('ing', 0.44), ('sensation', 0.215)]
Topic 3:
[('us', 167.807), ('majority', 164.525), ('thankfully', 159.259), ('syrup', 142.072), ('good', 110.156), ('easy', 104.675), ('till', 102.481)]
Topic 4:
[('kitchenaide', 0.441), ('earned', 0.441), ('recognizing', 0.389), ('coffee-cakes', 0.389), ('things-', 0.389), ('coffee-cake', 0.389), ('fragrant', 0.389)]






########## reviews for individual recipes (~50 reviews per recipe) #############
### group: 5, recipe_id: 18597 ###
Number of reviews in group 5: 49
Category distributions: 
 4    19
1    13
3    10
2     6
0     1
dtype: int64
Topic 0:
[('compliment', 1.204), ('husband', 1.19), ('supper', 1.187), ('having', 1.185), ('delicious', 1.183), ('easy', 1.182), ('recipie', 1.181)]
Topic 1:
[('rice', 10.222), ('thanks', 4.891), ('love', 4.879), ('mizznezz', 3.966), ('pilaf', 3.947), ('lot', 3.035), ('meal', 3.032)]
Topic 2:
[('rice', 7.637), ('dish', 6.721), ('celery', 4.88), ('cup', 4.874), ('pilaf', 3.047), ('broth', 3.039), ('herb', 3.036)]
Topic 3:
[('rice', 13.659), ('make', 5.643), ('thanks', 5.048), ('dish', 4.864), ('instead', 4.665), ('good', 4.442), ('did', 3.881)]
Topic 4:
[('added', 9.947), ('little', 8.683), ('t', 8.525), ('time', 7.995), ('pilaf', 7.593), ('stove', 7.46), ('didn', 6.688)]


### group: 2, recipe_id: 10155 ###
Number of reviews in group 2: 48
Category distributions: 
 0    46
3     1
4     1
dtype: int64
Topic 0:
[('rib', 36.637), ('pineapple', 24.89), ('juice', 24.884), ('tender', 21.824), ('easy', 21.193), ('thanks', 15.768), ('make', 15.274)]
Topic 1:
[('flavor', 0.296), ('pineapple', 0.294), ('juice', 0.292), ('keeper', 0.291), ('scampi', 0.291), ('sheet', 0.291), ('cook', 0.29)]
Topic 2:
[('section', 0.293), ('ok', 0.293), ('doubled', 0.29), ('rib', 0.289), ('oat', 0.289), ('thank-you', 0.288), ('thrilled', 0.288)]
Topic 3:
[('love', 1.537), ('-', 1.366), ('jim', 1.316), ('said', 1.282), ('turn', 1.201), ('passed', 1.2), ('mr', 1.193)]
Topic 4:
[('sweet', 1.211), ('happy', 1.194), ('felt', 1.194), ('needed', 1.183), ('sweetness', 1.18), ('juice', 0.38), ('pineapple', 0.34)]

### group: 3, recipe_id: 12436 ###
Number of reviews in group 3: 47
Category distributions: 
 0    12
2    10
4    10
1     9
3     6
dtype: int64
Topic 0:
[('used', 11.334), ('thanks', 5.807), ('really', 5.794), ('almond', 4.873), ('bean', 3.983), ('instead', 3.957), ('onion', 3.946)]
Topic 1:
[('bean', 6.719), ('oregano', 5.792), ('green', 5.78), ('used', 4.877), ('garlic', 4.875), ('thanks', 3.963), ('just', 3.944)]
Topic 2:
[('bean', 13.159), ('green', 8.551), ('onion', 5.783), ('pepper', 4.881), ('garlic', 4.873), ('nut', 4.866), ('used', 4.848)]
Topic 3:
[('fresh', 5.814), ('green', 5.809), ('t', 4.896), ('thyme', 4.887), ('bean', 4.88), ('good', 3.956), ('didn', 2.126)]
Topic 4:
[('bean', 8.567), ('used', 5.8), ('fresh', 5.792), ('nut', 4.868), ('served', 3.969), ('flavor', 3.951), ('thanks', 3.95)]


### group: 0, recipe_id: 4218 ###
Number of reviews in group 0: 47
Category distributions: 
 3    47
dtype: int64
Topic 0:
[('scrumptious', 0.296), ('sauce', 0.291), ('bread', 0.291), ('ginger', 0.291), ('lost', 0.288), ('sirloin', 0.287), ('shoot', 0.287)]
Topic 1:
[('oat', 0.299), ('loaf', 0.291), ('scampi', 0.291), ('sheet', 0.291), ('suggestion', 0.289), ('upped', 0.289), ('play', 0.288)]
Topic 2:
[('oat', 0.294), ('section', 0.293), ('ok', 0.293), ('thank-you', 0.288), ('thrilled', 0.288), ('noticed', 0.288), ('son', 0.286)]
Topic 3:
[('bread', 57.238), ('loaf', 20.439), ('flour', 19.548), ('used', 18.586), ('make', 17.722), ('wheat', 16.822), ('machine', 14.974)]
Topic 4:
[('bread', 0.366), ('live', 0.363), ('water', 0.356), ('used', 0.347), ('cup', 0.334), ('yeast', 0.332), ('uk', 0.328)]


