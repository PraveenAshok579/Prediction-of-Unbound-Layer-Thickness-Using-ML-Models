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
        self.root.title("Industrial ML – Unbound Thickness Prediction v1.0")
        self.root.geometry("1000x600")

        self.df = None
        self.results_df = None

        self.create_widgets()

    def create_widgets(self):

        frame_top = ttk.Frame(self.root)
        frame_top.pack(pady=15)

        ttk.Button(frame_top, text="Load Dataset", command=self.load_data).grid(row=0, column=0, padx=10)

        ttk.Label(frame_top, text="Select Target Column:").grid(row=0, column=1)

        self.target_combo = ttk.Combobox(frame_top, width=30)
        self.target_combo.grid(row=0, column=2, padx=10)

        ttk.Button(frame_top, text="Train & Compare Models", command=self.train_models).grid(row=0, column=3, padx=10)

        ttk.Button(frame_top, text="Export Results", command=self.export_results).grid(row=0, column=4, padx=10)

        # Table
        self.tree = ttk.Treeview(
            self.root,
            columns=("Model", "R2", "RMSE", "MAPE", "CV_R2"),
            show="headings"
        )
        self.tree.pack(fill="both", expand=True, pady=20)

        for col in ("Model", "R2", "RMSE", "MAPE", "CV_R2"):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=180)

    # ------------------------------------------------------

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)

                # Keep only numeric columns
                self.df = self.df.select_dtypes(include=np.number)

                self.target_combo['values'] = list(self.df.columns)

                messagebox.showinfo("Success",
                                    f"Dataset Loaded Successfully\nRows: {self.df.shape[0]}\nColumns: {self.df.shape[1]}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # ------------------------------------------------------

    def train_models(self):

        if self.df is None:
            messagebox.showerror("Error", "Please load dataset first")
            return

        target = self.target_combo.get()

        if target == "":
            messagebox.showerror("Error", "Please select target column")
            return

        X = self.df.drop(columns=[target])
        y = self.df[target]

        if X.shape[1] == 0:
            messagebox.showerror("Error", "No feature columns available")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor(objective='reg:squarederror', verbosity=0),
            "AdaBoost": AdaBoostRegressor()
        }

        results = []

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = mean_absolute_percentage_error(y_test, y_pred)

                cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()

                results.append([
                    name,
                    round(r2, 4),
                    round(rmse, 4),
                    round(mape, 4),
                    round(cv_r2, 4)
                ])

            except Exception as e:
                print(f"Error in {name}: {e}")

        self.results_df = pd.DataFrame(
            results,
            columns=["Model", "R2", "RMSE", "MAPE", "CV_R2"]
        )

        self.display_results()

    # ------------------------------------------------------

    def display_results(self):

        for row in self.tree.get_children():
            self.tree.delete(row)

        if self.results_df is not None:
            for _, row in self.results_df.iterrows():
                self.tree.insert("", "end", values=list(row))

    # ------------------------------------------------------

    def export_results(self):

        if self.results_df is None:
            messagebox.showerror("Error", "No results available to export")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")]
        )

        if file_path:
            try:
                self.results_df.to_excel(file_path, index=False)
                messagebox.showinfo("Success", "Results exported successfully")
            except Exception as e:
                messagebox.showerror("Error", str(e))


# ----------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = IndustrialMLApp(root)
    root.mainloop()
