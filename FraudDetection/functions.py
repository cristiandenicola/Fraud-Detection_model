import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
from matplotlib import gridspec

dataset = pd.read_csv("creditcard.csv")

##metodi usati per lo studio del dataset
#metodo di calcolo proporzione dei casi fraudolenti all'interno di quelli totali
def fraudCasesProp ():
    from colorama import Fore, Style
    #controlliamo le proporzioni tra casi di frode e casi validi
    print("Transazioni fraudolente: ", Fore.LIGHTRED_EX + str(len(dataset[dataset["Class"] == 1])) + Style.RESET_ALL)
    print("Transazioni valide: ", Fore.LIGHTGREEN_EX + str(len(dataset[dataset["Class"] == 0])) + Style.RESET_ALL)

    print("Proporzione della percentuale dei casi fraudolenti su quelli totali: {:.5f}".format(len(dataset[dataset["Class"] == 1])/dataset.shape[0]))   

    #rappresentazione grafica della proporzione
    data_p = dataset.copy()
    data_p[" "] = np.where(data_p["Class"] == 1, "Fraud", "Valid")

    data_p[" "].value_counts().plot(kind="pie")
    plt.ylabel("")
    plt.show()

#metodo di calcolo della distribuzione nel tempo delle transazioni e media dell'importo prelevato nelle transazioni con frode e senza
def timeTransactionsDistribution ():
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    time_value = dataset["Time"].values

    sns.histplot(time_value, kde=True, color="m", fill=True, ax=ax)
    ax.set(title='Distribution of transactions over Time')

    plt.show()

    #controllo se c'è differenza nel campo "Amount" tra le operazioni con frode e quelle valide
    print("Importo medio prelevato nelle transazione con frode: {:.5f}".format(dataset[dataset["Class"] == 1]["Amount"].mean()))
    print("Importo medio prelevato nelle transazione con valide: {:.5f}".format(dataset[dataset["Class"] == 0]["Amount"].mean()))

#studio della caratteristica del dataset 'Amount'
def amountValueDistribution ():
    print("Resoconto della caratteristica - Amount" + "\n-------------------------------")
    print(dataset["Amount"].describe())

#studio di tutte le caratteristiche presenti nel dataset
#in particolare si è studiata la sovrapposizione dei casi fraudolenti e casi validi per ciascuna feature del dataset
def generalFeaturesDistribution():
    data_plot = dataset.copy()
    
    #per motivi di organizzazione sposto la colonna Amount in prima posizione, seguir+à time, poi V1...
    amount = data_plot['Amount']
    data_plot.drop(labels=['Amount'], axis = 1, inplace = True)
    data_plot.insert(0, 'Amount', amount)

    # generaiamo gli istogrammi corrispondenti alle varie caratteristiche 
    columns = data_plot.iloc[:,0:30].columns
    plt.figure(figsize=(12,30*4))
    grids = gridspec.GridSpec(30, 1)

    for grid, index in enumerate(data_plot[columns]):
        ax = plt.subplot(grids[grid])
        sns.distplot(data_plot[index][data_plot.Class == 1], hist=False, kde_kws={"shade": True}, bins=50)
        sns.distplot(data_plot[index][data_plot.Class == 0], hist=False, kde_kws={"shade": True}, bins=50)
        ax.set_xlabel("")
        ax.set_title("Resoconto della caratteristica: "  + str(index))

    plt.show()

#metodo checker che controlla se all'interno del dataset sono presenti valori nulli
def nullValuesChecker():
    dataset.isnull().shape[0]
    print("Valori non nulli: " + str(dataset.isnull().shape[0]))
    print("Valori nulli: " + str(dataset.shape[0] - dataset.isnull().shape[0]))

## metodi usati per il modello e il suo train
# metodo usato per trainare il modello e restituisce le metriche. il modello viene salvato.
def modelTraining(X_train, y_train, X_test, y_test):
    # Import the classifiers
    from sklearn.ensemble import RandomForestClassifier

    # Import metrics
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    from colorama import Fore, Style

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    joblib.dump(rfc, 'model_rfc.pkl') ## salvataggio modello

    y_pred = rfc.predict(X_test)

    print("Tasso di accuratezza: ", Fore.LIGHTGREEN_EX + str(accuracy_score(y_test, y_pred)) + Style.RESET_ALL)
    ## metrica che misura la percentuale di predizioni corrette rispetto al numero totale di predizioni
    print("Tasso di precisione: ", Fore.LIGHTGREEN_EX + str(precision_score(y_test, y_pred)) + Style.RESET_ALL)
    ## metrica che misura la proporzione di predizioni positive corrette rispetto a tutte le predizioni positive effettuate
    print("Tasso di richiamo: ", Fore.LIGHTGREEN_EX + str(recall_score(y_test, y_pred)) + Style.RESET_ALL)
    ## metrica che misura la proporzione di campioni positivi correttamente predetti dal modello rispetto a tutti i campioni positivi effettivi
    print("F1: ", Fore.LIGHTGREEN_EX + str(f1_score(y_test, y_pred)) + Style.RESET_ALL)
    ## metrica che combina la precision e il recall in un unico valore.

# import del modello salvato
def modelTesting():
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler().fit(dataset[["Time", "Amount"]])
    dataset[["Time", "Amount"]] = scaler.transform(dataset[["Time", "Amount"]])

    from sklearn.model_selection import train_test_split
    y = dataset["Class"]
    X = dataset.iloc[:,0:30]
    ## X insieme delle features, input del modello
    ## y insieme delle targets, output che il modello andrà a prevedere
    ## test_size indica la proporzione dei dati tra train e test
    ## random_state seme di generazione di numeri random (impostato per non rendere tutto volatile)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)
    #modelTraining(X_train, X_test, y_train, y_test)
    loaded_model = joblib.load('model_rfc.pkl')
    y_pred = loaded_model.predict(X_test)

