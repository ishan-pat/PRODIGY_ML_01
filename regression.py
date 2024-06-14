# import tkinter as tk
# from tkinter import filedialog, messagebox, ttk
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.cluster import KMeans
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# import numpy as np

# class MLApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("ML Model Trainer")
#         self.root.geometry("800x600")
#         self.root.configure(bg="#e0f7fa")

#         # Frame for dataset selection
#         self.frame1 = tk.Frame(root, bg="#e0f7fa")
#         self.frame1.pack(pady=10, padx=20, fill=tk.X)

#         self.label1 = tk.Label(self.frame1, text="Select Dataset:", font=("Helvetica", 12), bg="#e0f7fa")
#         self.label1.pack(side=tk.LEFT, padx=5)

#         self.button1 = tk.Button(self.frame1, text="Browse", command=self.load_dataset, bg="#4db6ac", fg="white", font=("Helvetica", 10, "bold"))
#         self.button1.pack(side=tk.LEFT, padx=5)

#         self.dataset_path = tk.Entry(self.frame1, width=50, font=("Helvetica", 10))
#         self.dataset_path.pack(side=tk.LEFT, padx=5)

#         # Frame for model selection
#         self.frame2 = tk.Frame(root, bg="#e0f7fa")
#         self.frame2.pack(pady=10, padx=20, fill=tk.X)

#         self.label2 = tk.Label(self.frame2, text="Select Model:", font=("Helvetica", 12), bg="#e0f7fa")
#         self.label2.pack(side=tk.LEFT, padx=5)

#         self.model_var = tk.StringVar()
#         self.model_combo = ttk.Combobox(self.frame2, textvariable=self.model_var, font=("Helvetica", 10))
#         self.model_combo['values'] = ("Linear Regression", "K-means Clustering")
#         self.model_combo.current(0)
#         self.model_combo.pack(side=tk.LEFT, padx=5)

#         # Frame for buttons
#         self.frame3 = tk.Frame(root, bg="#e0f7fa")
#         self.frame3.pack(pady=10, padx=20, fill=tk.X)

#         self.train_button = tk.Button(self.frame3, text="Train Model", command=self.train_model, bg="#4db6ac", fg="white", font=("Helvetica", 10, "bold"))
#         self.train_button.pack(side=tk.LEFT, padx=5)

#         self.result_button = tk.Button(self.frame3, text="Show Results", command=self.show_results, bg="#4db6ac", fg="white", font=("Helvetica", 10, "bold"))
#         self.result_button.pack(side=tk.LEFT, padx=5)

#         self.results_text = tk.Text(root, height=15, width=80, font=("Courier", 10), wrap=tk.WORD, bg="#ffffff", fg="#000000", borderwidth=2, relief="sunken")
#         self.results_text.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
#         self.results_text.tag_configure("header", font=("Helvetica", 12, "bold"), foreground="#4caf50")
#         self.results_text.tag_configure("label", font=("Helvetica", 10, "bold"), foreground="#ff5722")
#         self.results_text.tag_configure("data", font=("Courier", 10), foreground="#000000")

#         self.dataset = None
#         self.model = None

#     def load_dataset(self):
#         file_selected = filedialog.askopenfilename()
#         self.dataset_path.delete(0, tk.END)
#         self.dataset_path.insert(0, file_selected)
#         self.dataset = pd.read_csv(file_selected)
#         messagebox.showinfo("Dataset", "Dataset Loaded Successfully!")

#     def train_model(self):
#         if self.dataset is None:
#             messagebox.showerror("Error", "No dataset loaded.")
#             return

#         model_choice = self.model_var.get()
#         if model_choice == "Linear Regression":
#             self.train_linear_regression()
#         elif model_choice == "K-means Clustering":
#             self.train_kmeans_clustering()

#     def train_linear_regression(self):
#         X = self.dataset.iloc[:, :-1]
#         y = self.dataset.iloc[:, -1]

#         # Convert categorical variables to numeric using One-Hot Encoding
#         X = pd.get_dummies(X, drop_first=True)

#         # Handle missing values by filling with mean of the column
#         X = X.fillna(X.mean())
#         y = y.fillna(y.mean())

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         self.model = LinearRegression()
#         self.model.fit(X_train, y_train)
#         y_pred = self.model.predict(X_test)

#         mse = mean_squared_error(y_test, y_pred)
#         mae = mean_absolute_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
        
#         self.results_text.delete(1.0, tk.END)
#         self.results_text.insert(tk.END, "Linear Regression Results\n", "header")
#         self.results_text.insert(tk.END, f"\nMean Squared Error: {mse:.2f}\n", "label")
#         self.results_text.insert(tk.END, f"Mean Absolute Error: {mae:.2f}\n", "label")
#         self.results_text.insert(tk.END, f"R^2 Score: {r2:.2f}\n", "label")

#         results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#         results_df.to_csv('linear_regression_results.csv', index=False)
#         messagebox.showinfo("CSV File", "Results saved to 'linear_regression_results.csv'.")

#     def train_kmeans_clustering(self):
#         X = self.dataset.values

#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         self.model = KMeans(n_clusters=2, random_state=42)
#         self.model.fit(X_scaled)
#         labels = self.model.labels_

#         self.results_text.delete(1.0, tk.END)
#         self.results_text.insert(tk.END, "K-means Clustering Results\n", "header")
#         self.results_text.insert(tk.END, classification_report(labels, labels, zero_division=1))

#         results_df = pd.DataFrame({'Data': range(len(labels)), 'Cluster': labels})
#         results_df.to_csv('kmeans_clustering_results.csv', index=False)
#         messagebox.showinfo("CSV File", "Results saved to 'kmeans_clustering_results.csv'.")

#     def show_results(self):
#         if self.model is None:
#             messagebox.showerror("Error", "No model trained.")
#             return

#         messagebox.showinfo("Results", "Results are shown in the text box below.")

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = MLApp(root)
#     root.mainloop()


import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Model Trainer")
        self.root.geometry("800x600")
        self.root.configure(bg="#e0f7fa")

        # Frame for dataset selection
        self.frame1 = tk.Frame(root, bg="#e0f7fa")
        self.frame1.pack(pady=10, padx=20, fill=tk.X)

        self.label1 = tk.Label(self.frame1, text="Select Dataset:", font=("Helvetica", 12), bg="#e0f7fa")
        self.label1.pack(side=tk.LEFT, padx=5)

        self.button1 = tk.Button(self.frame1, text="Browse", command=self.load_dataset, bg="#4db6ac", fg="white", font=("Helvetica", 10, "bold"))
        self.button1.pack(side=tk.LEFT, padx=5)

        self.dataset_path = tk.Entry(self.frame1, width=50, font=("Helvetica", 10))
        self.dataset_path.pack(side=tk.LEFT, padx=5)

        # Frame for model selection
        self.frame2 = tk.Frame(root, bg="#e0f7fa")
        self.frame2.pack(pady=10, padx=20, fill=tk.X)

        self.label2 = tk.Label(self.frame2, text="Select Model:", font=("Helvetica", 12), bg="#e0f7fa")
        self.label2.pack(side=tk.LEFT, padx=5)

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(self.frame2, textvariable=self.model_var, font=("Helvetica", 10))
        self.model_combo['values'] = ("Linear Regression", "K-means Clustering")
        self.model_combo.current(0)
        self.model_combo.pack(side=tk.LEFT, padx=5)

        # Frame for buttons
        self.frame3 = tk.Frame(root, bg="#e0f7fa")
        self.frame3.pack(pady=10, padx=20, fill=tk.X)

        self.train_button = tk.Button(self.frame3, text="Train Model", command=self.train_model, bg="#4db6ac", fg="white", font=("Helvetica", 10, "bold"))
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.result_button = tk.Button(self.frame3, text="Show Results", command=self.show_results, bg="#4db6ac", fg="white", font=("Helvetica", 10, "bold"))
        self.result_button.pack(side=tk.LEFT, padx=5)

        self.results_text = tk.Text(root, height=15, width=80, font=("Courier", 10), wrap=tk.WORD, bg="#ffffff", fg="#000000", borderwidth=2, relief="sunken")
        self.results_text.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.results_text.tag_configure("header", font=("Helvetica", 12, "bold"), foreground="#4caf50")
        self.results_text.tag_configure("label", font=("Helvetica", 10, "bold"), foreground="#ff5722")
        self.results_text.tag_configure("data", font=("Courier", 10), foreground="#000000")

        self.dataset = None
        self.model = None

    def load_dataset(self):
        file_selected = filedialog.askopenfilename()
        self.dataset_path.delete(0, tk.END)
        self.dataset_path.insert(0, file_selected)
        self.dataset = pd.read_csv(file_selected)
        messagebox.showinfo("Dataset", "Dataset Loaded Successfully!")

    def train_model(self):
        if self.dataset is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return

        model_choice = self.model_var.get()
        if model_choice == "Linear Regression":
            self.train_linear_regression()
        elif model_choice == "K-means Clustering":
            self.train_kmeans_clustering()

    def train_linear_regression(self):
        X = self.dataset.iloc[:, :-1]
        y = self.dataset.iloc[:, -1]

        # Convert categorical variables to numeric using One-Hot Encoding
        X = pd.get_dummies(X, drop_first=True)

        # Handle missing values by filling with mean of the column
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Linear Regression Results\n", "header")
        self.results_text.insert(tk.END, f"\nMean Squared Error: {mse:.2f}\n", "label")
        self.results_text.insert(tk.END, f"Mean Absolute Error: {mae:.2f}\n", "label")
        self.results_text.insert(tk.END, f"R^2 Score: {r2:.2f}\n", "label")

        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        results_df.to_csv('linear_regression_results.csv', index=False)
        messagebox.showinfo("CSV File", "Results saved to 'linear_regression_results.csv'.")

    def train_kmeans_clustering(self):
        X = self.dataset

        # Convert categorical variables to numeric using One-Hot Encoding
        X = pd.get_dummies(X, drop_first=True)

        # Handle missing values by filling with mean of the column
        X = X.fillna(X.mean())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.model = KMeans(n_clusters=2, random_state=42)
        self.model.fit(X_scaled)
        labels = self.model.labels_

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "K-means Clustering Results\n", "header")
        self.results_text.insert(tk.END, classification_report(labels, labels, zero_division=1))

        results_df = pd.DataFrame({'Data': range(len(labels)), 'Cluster': labels})
        results_df.to_csv('kmeans_clustering_results.csv', index=False)
        messagebox.showinfo("CSV File", "Results saved to 'kmeans_clustering_results.csv'.")

    def show_results(self):
        if self.model is None:
            messagebox.showerror("Error", "No model trained.")
            return

        messagebox.showinfo("Results", "Results are shown in the text box below.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()
