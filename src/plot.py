
import matplotlib.pyplot as plt
    
def plot_loss(loss_train, loss_test = None, title= "Evolución del error cuadrático"):
    epochs = list(range(1, len(loss_train) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss_train, label="Train", color="teal", linewidth=2)
    if loss_test is not None:
        plt.plot(epochs, loss_test, label="Test", color="indianred", linestyle="--", linewidth=2)
        
    plt.title(title, fontsize= 13)
    plt.xlabel("Época", fontsize= 12)
    plt.ylabel("Loss (EC)", fontsize= 12)
    plt.legend(fontsize= 13)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    plt.show()

colors_named = [
    "Crimson", "DarkOrange", "Goldenrod", "RoyalBlue", "ForestGreen",
    "MediumVioletRed", "DarkTurquoise", "SlateBlue", "DarkGoldenrod", "Firebrick"
]



def plot_train_test_metrics(results, plot_train = False, plot_test = True):
    # --- Loss ---
    plt.figure(figsize=(8, 5))
    for idx, (alpha, hist) in enumerate(results.items()):
        color = colors_named[idx % len(colors_named)]
        epochs = range(1, len(hist['loss_train']) + 1)
        if plot_train:
            plt.plot(epochs, hist['loss_train'], label=f"Train loss (α={alpha})", color=color)
        if plot_test:
            plt.plot(epochs, hist['loss_test'], linestyle='--' if plot_train else '-', label=f"Test loss (α={alpha})", color=color)

    plt.title("Pérdida Error Cuadrático: Train vs Test", fontsize= 16)
    plt.xlabel("Épocas", fontsize= 14)
    plt.ylabel("SE (Squared Errors)", fontsize= 14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    # --- Accuracy ---
    plt.figure(figsize=(8, 5))
    for idx, (alpha, hist) in enumerate(results.items()):
        epochs = range(1, len(hist['acc_train']) + 1)
        color = colors_named[idx + 5 % len(colors_named)]
        if plot_train:
            plt.plot(epochs, hist['acc_train'], label=f"Train acc (α={alpha})", color=color)
        if plot_test:
            plt.plot(epochs, hist['acc_test'], linestyle='--' if plot_train else '-', label=f"Test  acc (α={alpha})", color=color)
    plt.title("Accuracy: Train vs Test", fontsize= 16)
    plt.xlabel("Épocas", fontsize= 14)
    plt.ylabel("Accuracy", fontsize= 14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10)
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