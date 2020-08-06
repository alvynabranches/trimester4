from sklearn.svm import SVC, LinearSVC
from pandas import DataFrame

score_matrix = DataFrame(columns=['kernel', 'C', 'degree', 'score'])

# 2 dimensions
print('2 dimensions')
x1 = [True, True, False, False]
x2 = [True, False, True, False]
y_and = [(_x1 and _x2) for _x1, _x2, in zip(x1, x2)]
y_or = [(_x1 or _x2) for _x1, _x2, in zip(x1, x2)]
data = DataFrame(data=dict(x1=x1, x2=x2, y_and=y_and, y_or=y_or))
print('Linear')
model_and = SVC(kernel='linear', C=1.0)
model_or = SVC(kernel='linear', C=1.0)
model_and.fit(data[['x1', 'x2']], data['y_and'])
model_or.fit(data[['x1', 'x2']], data['y_or'])
print(f"AND: Score = {model_and.score(data[['x1', 'x2']], data['y_and'])}, Intercept = {model_and.intercept_}, Coefficent = {model_and.coef_}")
print(f"OR: Score = {model_or.score(data[['x1', 'x2']], data['y_or'])}, Intercept = {model_or.intercept_}, Coefficent = {model_or.coef_}")
# score_matrix = score_matrix.append()
print()
print('Radial')
model_and = SVC(kernel='rbf', C=1.0)
model_or = SVC(kernel='rbf', C=1.0)
model_and.fit(data[['x1', 'x2']], data['y_and'])
model_or.fit(data[['x1', 'x2']], data['y_or'])
print(f"AND: Score = {model_and.score(data[['x1', 'x2']], data['y_and'])}")
print(f"OR: Score = {model_or.score(data[['x1', 'x2']], data['y_or'])}")
print()
print('Polynomial - Degree = 3')
model_and = SVC(kernel='poly', C=1.0, degree=3)
model_or = SVC(kernel='poly', C=1.0, degree=3)
model_and.fit(data[['x1', 'x2']], data['y_and'])
model_or.fit(data[['x1', 'x2']], data['y_or'])
print(f"AND: {model_and.score(data[['x1', 'x2']], data['y_and'])}")
print(f"OR: {model_or.score(data[['x1', 'x2']], data['y_or'])}")

print()
print()
print()
# 3 dimensions
print('3 dimensions')
x1 = [True, True, True, True, False, False, False, False]
x2 = [True, True, False, False, True, True, False, False]
x3 = [True, False, True, False, True, False, True, False]
y_and = [(_x1 and _x2 and _x3) for _x1, _x2, _x3 in zip(x1, x2, x3)]
y_or = [(_x1 or _x2 or _x3) for _x1, _x2, _x3 in zip(x1, x2, x3)]
data = DataFrame(data=dict(x1=x1, x2=x2, x3=x3, y_and=y_and, y_or=y_or))
print('Linear')
model_and = SVC(kernel='linear', C=1.0)
model_or = SVC(kernel='linear', C=1.0)
model_and.fit(data[['x1', 'x2', 'x3']], data['y_and'])
model_or.fit(data[['x1', 'x2', 'x3']], data['y_or'])
print(f"AND: {model_and.score(data[['x1', 'x2', 'x3']], data['y_and'])}, Intercept = {model_and.intercept_}, Coefficent = {model_and.coef_}")
print(f"OR: {model_or.score(data[['x1', 'x2', 'x3']], data['y_or'])}, Intercept = {model_or.intercept_}, Coefficent = {model_or.coef_}")
print()
print('Radial')
model_and = SVC(kernel='rbf', C=1.0)
model_or = SVC(kernel='rbf', C=1.0)
model_and.fit(data[['x1', 'x2', 'x3']], data['y_and'])
model_or.fit(data[['x1', 'x2', 'x3']], data['y_or'])
print(f"AND: {model_and.score(data[['x1', 'x2', 'x3']], data['y_and'])}")
print(f"OR: {model_or.score(data[['x1', 'x2', 'x3']], data['y_or'])}")
print()
print('Polynomial - Degree = 3')
model_and = SVC(kernel='poly', C=1.0, degree=3)
model_or = SVC(kernel='poly', C=1.0, degree=3)
model_and.fit(data[['x1', 'x2', 'x3']], data['y_and'])
model_or.fit(data[['x1', 'x2', 'x3']], data['y_or'])
print(f"AND: {model_and.score(data[['x1', 'x2', 'x3']], data['y_and'])}")
print(f"OR: {model_or.score(data[['x1', 'x2', 'x3']], data['y_or'])}")
print()
print('Polynomial - Degree = 4')
model_and = SVC(kernel='poly', C=1.0, degree=4)
model_or = SVC(kernel='poly', C=1.0, degree=4)
model_and.fit(data[['x1', 'x2', 'x3']], data['y_and'])
model_or.fit(data[['x1', 'x2', 'x3']], data['y_or'])
print(f"AND: {model_and.score(data[['x1', 'x2', 'x3']], data['y_and'])}")
print(f"OR: {model_or.score(data[['x1', 'x2', 'x3']], data['y_or'])}")

print()
print()
print()
# 4 dimensions
print('4 dimensions')
x1 = [True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False]
x2 = [True, True, True, True, False, False, False, False, True, True, True, True, False, False, False, False]
x3 = [True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False]
x4 = [True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False]
y_and = [(_x1 and _x2 and _x3 and _x4) for _x1, _x2, _x3, _x4 in zip(x1, x2, x3, x4)]
y_or = [(_x1 or _x2 or _x3 or _x4) for _x1, _x2, _x3, _x4 in zip(x1, x2, x3, x4)]
data = DataFrame(data=dict(x1=x1, x2=x2, x3=x3, x4=x4, y_and=y_and, y_or=y_or))
print('Linear')
model_and = SVC(kernel='linear', C=1.0)
model_or = SVC(kernel='linear', C=1.0)
model_and.fit(data[['x1', 'x2', 'x3', 'x4']], data['y_and'])
model_or.fit(data[['x1', 'x2', 'x3', 'x4']], data['y_or'])
print(f"AND: {model_and.score(data[['x1', 'x2', 'x3', 'x4']], data['y_and'])}, Intercept = {model_and.intercept_}, Coefficent = {model_and.coef_}")
print(f"OR: {model_or.score(data[['x1', 'x2', 'x3', 'x4']], data['y_or'])}, Intercept = {model_or.intercept_}, Coefficent = {model_or.coef_}")
print()
print('Radial')
model_and = SVC(kernel='rbf', C=1.0)
model_or = SVC(kernel='rbf', C=1.0)
model_and.fit(data[['x1', 'x2', 'x3', 'x4']], data['y_and'])
model_or.fit(data[['x1', 'x2', 'x3', 'x4']], data['y_or'])
print(f"AND: {model_and.score(data[['x1', 'x2', 'x3', 'x4']], data['y_and'])}")
print(f"OR: {model_or.score(data[['x1', 'x2', 'x3', 'x4']], data['y_or'])}")
print()
print('Polynomial - Degree = 3')
model_and = SVC(kernel='poly', C=1.0, degree=3)
model_or = SVC(kernel='poly', C=1.0, degree=3)
model_and.fit(data[['x1', 'x2', 'x3', 'x4']], data['y_and'])
model_or.fit(data[['x1', 'x2', 'x3', 'x4']], data['y_or'])
print(f"AND: {model_and.score(data[['x1', 'x2', 'x3', 'x4']], data['y_and'])}")
print(f"OR: {model_or.score(data[['x1', 'x2', 'x3', 'x4']], data['y_or'])}")
print('Polynomial - Degree = 4')
model_and = SVC(kernel='poly', C=1.0, degree=4)
model_or = SVC(kernel='poly', C=1.0, degree=4)
model_and.fit(data[['x1', 'x2', 'x3', 'x4']], data['y_and'])
model_or.fit(data[['x1', 'x2', 'x3', 'x4']], data['y_or'])
print(f"AND: {model_and.score(data[['x1', 'x2', 'x3', 'x4']], data['y_and'])}")
print(f"OR: {model_or.score(data[['x1', 'x2', 'x3', 'x4']], data['y_or'])}")

print()
print()
print()
# 5 dimensions
print('5 dimensions')
x1 = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
x2 = [True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False]
x3 = [True, True, True, True, False, False, False, False, True, True, True, True, False, False, False, False, True, True, True, True, False, False, False, False, True, True, True, True, False, False, False, False]
x4 = [True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False]
x5 = [True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False]
y_and = [(_x1 and _x2 and _x3 and _x4 and _x5) for _x1, _x2, _x3, _x4, _x5 in zip(x1, x2, x3, x4, x5)]
y_or = [(_x1 or _x2 or _x3 or _x4 or _x5) for _x1, _x2, _x3, _x4, _x5 in zip(x1, x2, x3, x4, x5)]
data = DataFrame(data=dict(x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, y_and=y_and, y_or=y_or))
print('Linear')
model_and = SVC(kernel='linear', C=1.0)
model_or = SVC(kernel='linear', C=1.0)
model_and.fit(data[['x1', 'x2', 'x3', 'x4', 'x5']], data['y_and'])
model_or.fit(data[['x1', 'x2', 'x3', 'x4', 'x5']], data['y_or'])
print(f"AND: {model_and.score(data[['x1', 'x2', 'x3', 'x4', 'x5']], data['y_and'])}, Intercept = {model_and.intercept_}, Coefficent = {model_and.coef_}")
print(f"OR: {model_or.score(data[['x1', 'x2', 'x3', 'x4', 'x5']], data['y_or'])}, Intercept = {model_or.intercept_}, Coefficent = {model_or.coef_}")
print()
print('Radial')
model_and = SVC(kernel='rbf', C=1.0)
model_or = SVC(kernel='rbf', C=1.0)
model_and.fit(data[['x1', 'x2', 'x3', 'x4', 'x5']], data['y_and'])
model_or.fit(data[['x1', 'x2', 'x3', 'x4', 'x5']], data['y_or'])
print(f"AND: {model_and.score(data[['x1', 'x2', 'x3', 'x4', 'x5']], data['y_and'])}")
print(f"OR: {model_or.score(data[['x1', 'x2', 'x3', 'x4', 'x5']], data['y_or'])}")
print()
print('Polynomial - Degree = 3')
model_and = SVC(kernel='poly', C=1.0, degree=3)
model_or = SVC(kernel='poly', C=1.0, degree=3)
model_and.fit(data[['x1', 'x2', 'x3', 'x4', 'x5']], data['y_and'])
model_or.fit(data[['x1', 'x2', 'x3', 'x4', 'x5']], data['y_or'])
print(f"AND: {model_and.score(data[['x1', 'x2', 'x3', 'x4', 'x5']], data['y_and'])}")
print(f"OR: {model_or.score(data[['x1', 'x2', 'x3', 'x4', 'x5']], data['y_or'])}")

print()
print()
print()
# 6 dimensions
print('6 dimensions')
x1 = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
x2 = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
x3 = [True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False]
x4 = [True, True, True, True, False, False, False, False, True, True, True, True, False, False, False, False, True, True, True, True, False, False, False, False, True, True, True, True, False, False, False, False, True, True, True, True, False, False, False, False, True, True, True, True, False, False, False, False, True, True, True, True, False, False, False, False, True, True, True, True, False, False, False, False]
x5 = [True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False]
x6 = [True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False]
y_and = [(_x1 and _x2 and _x3 and _x4 and _x5 and _x6) for _x1, _x2, _x3, _x4, _x5, _x6 in zip(x1, x2, x3, x4, x5, x6)]
y_or = [(_x1 or _x2 or _x3 or _x4 or _x5 or _x6) for _x1, _x2, _x3, _x4, _x5, _x6 in zip(x1, x2, x3, x4, x5, x6)]
data = DataFrame(data=dict(x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, x6=x6, y_and=y_and, y_or=y_or))
print('Linear')
model_and = SVC(kernel='linear', C=1.0)
model_or = SVC(kernel='linear', C=1.0)
model_and.fit(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']], data['y_and'])
model_or.fit(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']], data['y_or'])
print(f"AND: {model_and.score(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']], data['y_and'])}, Intercept = {model_and.intercept_}, Coefficent = {model_and.coef_}")
print(f"OR: {model_or.score(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']], data['y_or'])}, Intercept = {model_or.intercept_}, Coefficent = {model_or.coef_}")
print()
print('Radial')
model_and = SVC(kernel='rbf', C=1.0)
model_or = SVC(kernel='rbf', C=1.0)
model_and.fit(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']], data['y_and'])
model_or.fit(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']], data['y_or'])
print(f"AND: {model_and.score(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']], data['y_and'])}")
print(f"OR: {model_or.score(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']], data['y_or'])}")
print()
print('Polynomial - Degree = 3')
model_and = SVC(kernel='poly', C=1.0, degree=3)
model_or = SVC(kernel='poly', C=1.0, degree=3)
model_and.fit(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']], data['y_and'])
model_or.fit(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']], data['y_or'])
print(f"AND: {model_and.score(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']], data['y_and'])}")
print(f"OR: {model_or.score(data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']], data['y_or'])}")