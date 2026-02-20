import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor

class IndustrialMLApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Industrial ML – Unbound Thickness Prediction")
        self.root.geometry("950x600")

        self.df = None
        self.results_df = None

        self.create_widgets()

    def create_widgets(self):

        frame_top = ttk.Frame(self.root)
        frame_top.pack(pady=10)

        ttk.Button(frame_top, text="Load Dataset", command=self.load_data).grid(row=0, column=0, padx=10)
        ttk.Label(frame_top, text="Select Target:").grid(row=0, column=1)

        self.target_combo = ttk.Combobox(frame_top, width=25)
        self.target_combo.grid(row=0, column=2, padx=10)

        ttk.Button(frame_top, text="Train & Compare Models", command=self.train_models).grid(row=0, column=3, padx=10)
        ttk.Button(frame_top, text="Export Results", command=self.export_results).grid(row=0, column=4, padx=10)

        self.tree = ttk.Treeview(self.root, columns=("Model", "R2", "RMSE", "MAPE", "CV_R2"), show="headings")
        self.tree.pack(fill="both", expand=True, pady=20)

        for col in ("Model", "R2", "RMSE", "MAPE", "CV_R2"):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            self.target_combo['values'] = list(self.df.columns)
            messagebox.showinfo("Success", "Dataset Loaded Successfully")

    def train_models(self):
        if self.df is None:
            messagebox.showerror("Error", "Load dataset first")
            return

        target = self.target_combo.get()
        if target == "":
            messagebox.showerror("Error", "Select target column")
            return

        X = self.df.drop(columns=[target])
        y = self.df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor(objective='reg:squarederror'),
            "AdaBoost": AdaBoostRegressor()
        }

        results = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred)

            cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()

            results.append([name, r2, rmse, mape, cv_r2])

        self.results_df = pd.DataFrame(results, columns=["Model", "R2", "RMSE", "MAPE", "CV_R2"])

        self.display_results()

    def display_results(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        for _, row in self.results_df.iterrows():
            self.tree.insert("", "end", values=list(row))

    def export_results(self):
        if self.results_df is None:
            messagebox.showerror("Error", "No results to export")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx")
        if file_path:
            self.results_df.to_excel(file_path, index=False)
            messagebox.showinfo("Exported", "Results exported successfully")

if __name__ == "__main__":
    root = tk.Tk()
    app = IndustrialMLApp(root)
    root.mainloop()
