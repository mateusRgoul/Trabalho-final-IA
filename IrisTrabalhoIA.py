from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Carregar o conjunto de dados Iris
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Pré-processamento de Dados - Normalização
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Aprendizado Supervisionado - Classificação com Naive Bayes
classifier = GaussianNB()
scores = cross_val_score(classifier, X_normalized, y, cv=5)

# Avaliação de Desempenho - Precisão do modelo
print("Precisão média:", scores.mean())