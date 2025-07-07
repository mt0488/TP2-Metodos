
import matplotlib.pyplot as plt

def plot_loss(loss_history_train):
    plt.plot(loss_history_train)
    plt.xlabel("Época")
    plt.ylabel("Pérdida cuadrática")
    plt.title("Evolución de la pérdida durante el entrenamiento")
    plt.grid(True)
    plt.show()
    
def plot_loss_train_test(loss_train, loss_test):
    epochs = list(range(1, len(loss_train) + 1))
    plt.plot(epochs, loss_train, label="Train", color="blue")
    plt.plot(epochs, loss_test, label="Test", color="red", linestyle="--")
    plt.title("Error cuadrático: Train vs Test")
    plt.xlabel("Época")
    plt.ylabel("Loss (EC)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_train_test_metrics(results, plot_train = False, plot_test = True):
    """
    Grafica:
    1. Evolución de la pérdida (loss) en train y test para cada alpha.
    2. Evolución de la accuracy en train y test para cada alpha.
    """

    # --- Loss ---
    plt.figure(figsize=(8, 5))
    for alpha, hist in results.items():
        epochs = range(1, len(hist['loss_train']) + 1)
        if plot_train:
            plt.plot(epochs, hist['loss_train'], label=f"Train loss (α={alpha})")
        if plot_test:
            plt.plot(epochs, hist['loss_test'], linestyle='--', label=f"Test  loss (α={alpha})")
    plt.title("Pérdida SE: Train vs Test", fontsize= 16)
    plt.xlabel("Épocas", fontsize= 14)
    plt.ylabel("SE (Squared Errors)", fontsize= 14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid(True)

    # --- Accuracy ---
    plt.figure(figsize=(8, 5))
    for alpha, hist in results.items():
        epochs = range(1, len(hist['acc_train']) + 1)
        if plot_train:
            plt.plot(epochs, hist['acc_train'], label=f"Train acc (α={alpha})")
        if plot_test:
            plt.plot(epochs, hist['acc_test'], linestyle='--', label=f"Test  acc (α={alpha})")
    plt.title("Accuracy: Train vs Test", fontsize= 16)
    plt.xlabel("Épocas", fontsize= 14)
    plt.ylabel("Accuracy", fontsize= 14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.show()
    
    
def plot_confusion_matrix(cm, labels=["Healthy", "Parkinson"], titulo="Matriz de Confusión"):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel("Predicho", fontsize=14)
    ax.set_ylabel("Real", fontsize=14)
    ax.set_title(titulo, fontsize=16)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=16)

    plt.colorbar(im)
    plt.show()
    
    
def plot_metrics_by_size(results_by_size):

    sizes = sorted(results_by_size.keys())

    acc_train = [results_by_size[s]['acc_train_final'] for s in sizes]
    acc_test  = [results_by_size[s]['acc_test_final']  for s in sizes]
    mse_train = [results_by_size[s]['mse_train'] for s in sizes]
    mse_test  = [results_by_size[s]['mse_test']  for s in sizes]

    # --- Accuracy ---
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, acc_train, marker='o', label="Train Accuracy")
    plt.plot(sizes, acc_test, marker='o', linestyle='--', label="Test Accuracy")
    plt.title("Accuracy final según tamaño de imagen", fontsize=16)
    plt.xlabel("Tamaño de imagen", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(sizes, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend()
    
    # --- Pérdida (SSE) ---
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, mse_train, marker='o', label="Train SSE")
    plt.plot(sizes, mse_test, marker='o', linestyle='--', label="Test SSE")
    plt.title("SSE final según tamaño de imagen", fontsize=16)
    plt.xlabel("Tamaño de imagen", fontsize=14)
    plt.ylabel("Sum of Squared Errors", fontsize=14)
    plt.xticks(sizes, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend()
    
    plt.show()