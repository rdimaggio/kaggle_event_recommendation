import numpy as np
import csv as csv
from datetime import datetime
import pylab as pl
#from scipy.stats import spearmanr
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.linear_model import ElasticNet
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import auc_score
#from sklearn.preprocessing import normalize
from sklearn.utils import check_arrays
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_classif, chi2, RFECV
from sklearn.svm import SVC, SVR, LinearSVC, NuSVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import zero_one


class DataLoader():

    def __init__(self):
        pass

    def load_users(self, users_file):
        csvfile = csv.reader(open(users_file, 'rU'))
        header = csvfile.next()
        self.users = {}
        for row in csvfile:
            # locale
            # birthyear
            try:
              birthyear = int(row[2])
            except:
              birthyear = None
            # age
            age = 2013 - int(row[2]) if birthyear else None
            # gender
            male = True if row[3] == "male" else False
            # joined
            # user age
            try:
                user_age = (datetime.now() - datetime.strptime(row[4],
                                            "%Y-%m-%dT%H:%M:%S.%fZ")).days
            except:
                user_age = None
            # location
            # timezone
            timezone = int(row[6]) if row[6] else None
            self.users[row[0]] = {'locale': row[1], 'birthyear': birthyear,
                                  'age': age, 'gender': male, 'joined': row[4],
                                  'user_age': user_age, 'location': row[5],
                                  'timezone': timezone}


    def load_events(self, events_file):
        csvfile = csv.reader(open(events_file, 'rU'))
        header = csvfile.next()
        self.events = {}
        for row in csvfile:
            set_of_events = row[0].split(" ")
            for each in set_of_events:
                yes_attendees = row[1].split(" ")
                maybe_attendees = row[2].split(" ")
                invited_attendees = row[3].split(" ")
                no_attendees = row[4].split(" ")
                self.events[each] = {'yes': yes_attendees,
                                     'maybe': maybe_attendees,
                                     'invited': invited_attendees,
                                     'no': no_attendees}

    def load_interests(self, interests_file):
        csvfile = csv.reader(open(interests_file, 'rU'))
        header = csvfile.next()
        self.interests = []
        for row in csvfile:
            # user
            # event
            # invited
            invited = True if row[2] == 1 else False
            # timestamp
            try:
                timestamp = datetime.strptime(row[3].replace("+00:00",""),
                                          "%Y-%m-%d %H:%M:%S.%f")
            except:
                try:
                    timestamp = datetime.strptime(row[3].replace("+00:00",""),
                                              "%Y-%m-%d %H:%M:%S")
                except:
                    timestamp = None
            # interested
            interested = True if row[2] == 1 else False
            # not_interested
            not_interested = True if row[2] == 1 else False
            self.interests.append([row[0], row[1], invited, timestamp,
                                  interested, not_interested])

    def consolidate_data(self, output_file):
        with open(output_file, 'wb') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            for each in self.interests:
                user_info = self.users[each[0]]
                each.append(user_info['locale'])
                each.append(user_info['birthyear'])
                each.append(user_info['age'])
                each.append(user_info['gender'])
                each.append(user_info['joined'])
                each.append(user_info['user_age'])
                each.append(user_info['location'])
                each.append(user_info['timezone'])

                event_info = self.events[each[1]]
                each.append(True) if each[0] in event_info['yes'] \
                else each.append(False)
                each.append(True) if each[0] in event_info['maybe'] \
                else each.append(False)
                each.append(True) if each[0] in event_info['invited'] \
                else each.append(False)
                each.append(True) if each[0] in event_info['no'] \
                else each.append(False)

                csvwriter.writerow(each)

    def load_user_data(self, output_file, interests_file, user_file,
                       attendees_file):

        self.load_users(user_file)
        self.load_events(attendees_file)
        self.load_interests(interests_file)
        self.consolidate_data(output_file)


if __name__ == '__main__':

    USER_FILE = 'data/users.csv'
    USER_INTERESTS_FILE = 'data/train.csv'
    ATTENDEES_FILE = 'data/event_attendees.csv'
    EVENTS_FILE = 'data/events.csv'
    FRIENDS_FILE = 'data/user_friends.csv'
    USER_ACTIONS_FILE = 'data/compilations/user_actions.csv'

    start = datetime.now()
    dl = DataLoader()
    dl.load_user_data(USER_ACTIONS_FILE, USER_INTERESTS_FILE, USER_FILE,
                      ATTENDEES_FILE)
    stop = datetime.now()

    print "run time: " + str(stop - start)

"""def wmae(y_true, y_pred, weights):
    y_true, y_pred = check_arrays(y_true, y_pred)
    return (1 / np.sum(weights)) * (np.sum(weights * (y_pred - y_true)))


def replace_NA(row):
    for each in range(len(row)):
        row[each] = row[each].replace("NA", "0")
        row[each] = "0" if row[each] == "" else row[each]
    return row


def string_find_code(item, sub):
    for i in range(len(sub)):
        if item.find(sub[i]) >= 0:
            return float(i + 1)
    return 0.


def name_length(name):
    try:
        index = name.index(" (")
        return float(len(name[:index - 1].split(" ")))
    except:
        return float(len(name.split(" ")))


def convert_currency(row, items=[]):
    for each in items:
        row[each] = row[each].replace("$", "").replace(",", "")
    return row


def dual_cross_val_score(estimator1, estimator2, X, y, score_func,
                         train, test, verbose, ratio):
    "Inner loop for cross validation"

    estimator1.fit(X[train], y[train])
    estimator2.fit(X[train], y[train])

    guess = ratio * estimator1.predict(X[test]) + (1 - ratio) * \
        estimator2.predict(X[test])
    guess[guess < 0.5] = 0.
    guess[guess >= 0.5] = 1.
    score = score_func(y[test], guess)

    if verbose > 1:
        print("score: %f" % score)
    return score


def Bootstrap_cv(estimator1, estimator2, X, y, score_func, cv=None, n_jobs=1,
                 verbose=0, ratio=.5):
    X, y = cross_validation.check_arrays(X, y, sparse_format='csr')
    cv = cross_validation.check_cv(cv, X, y,
                                   classifier=
                                   cross_validation.is_classifier(estimator1))
    if score_func is None:
        if not hasattr(estimator1, 'score') or \
                not hasattr(estimator2, 'score'):
            raise TypeError(
                "If no score_func is specified, the estimator passed "
                "should have a 'score' method. The estimator %s "
                "does not." % estimator1)
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    scores = \
        cross_validation.Parallel(
            n_jobs=n_jobs, verbose=verbose)(
                cross_validation.delayed(
                    dual_cross_val_score)
                (cross_validation.clone(estimator1),
                 cross_validation.clone(estimator2),
                 X, y, score_func, train, test, verbose, ratio)
                for train, test in cv)
    return np.array(scores)

# load data
csv_file_object = csv.reader(open('overfitting.csv', 'rb'))
header = csv_file_object.next()
all_data = []
for row in csv_file_object:
    all_data.append(row)
all_data = np.array(all_data)
all_data = all_data.astype(np.float)

cutoff = 250
components = 113

# create each data set to use
# all data
all_y_practice = all_data[0::, 2]
all_y_leaderboard = all_data[0::, 3]
all_x = np.delete(all_data, [0, 1, 2, 3, 4], 1)

# train data
train_data = all_data[:cutoff]
train_y_practice = train_data[0::, 2]
train_y_leaderboard = train_data[0::, 3]
train_x = np.delete(train_data, [0, 1, 2, 3, 4], 1)

# test data
test_data = all_data[cutoff:]
entries = test_data[0::, 0]
test_y_practice = test_data[0::, 2]
test_x = np.delete(test_data, [0, 1, 2, 3, 4], 1)"""
