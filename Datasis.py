import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io


class  Datasis:
    def __init__(self, data=None):
        """
        Initialize the data analysis library with optional data.

        Args:
            data (DataFrame, optional): The initial dataset to work with. Defaults to None.
        """
        self.data = data
    
    def load_data(self, file_path):
        """
        Load data from a CSV, Excel, or JSON file.

        Args:
            file_path (str): The path to the file to be loaded.

        Returns:
            DataFrame or str: The loaded data or an error message.
        """
        ext = os.path.splitext(file_path)[1]
        try:
            if ext == '.csv':
                self.data = pd.read_csv(file_path)
            elif ext in ['.xls', '.xlsx']:
                self.data = pd.read_excel(file_path)
            elif ext == '.json':
                self.data = pd.read_json(file_path)
            else:
                raise ValueError("File format not supported")
        except Exception as e:
            return f"Error occurred while loading data: {e}"
        return self.data
    def DataFrame(self):
        """
        Get the current DataFrame.

        Returns:
            DataFrame or str: The current DataFrame or a message indicating no data.
        """
        if self.data is not None:
            return self.data
        return "No data available."
    def set_data(self, data):
        """
        Set the data for analysis.

        Args:
            data (DataFrame): The dataset to be set.
        """
        self.data = data
    def get_data(self):
        """
        Get the current dataset.

        Returns:
            DataFrame or str: The current dataset or a message indicating no data.
        """
        if self.data is not None:
            return self.data
        return "No data available."
    def show_data(self, n=5):
        """
        Display the first n rows of the data.

        Args:
            n (int): The number of rows to display. Default is 5.
        """
        if self.data is not None:
            return self.data.head(n)
        return "No data available for analysis."
    def tail(self, n=5):
        """
        Display the last n rows of the data.

        Args:
            n (int): The number of rows to display. Default is 5.
        """ 
        if self.data is not None:
            return self.data.tail(n)
        return "No data available for analysis."
    def head(self, n=5):
        """
        Display a random sample of n rows from the data.

        Args:
            n (int): The number of rows to display. Default is 5.
        """
        if self.data is not None:
            return self.data.sample(n)
        return "No data available for analysis."
    
    def summary(self):
        """
        Display a quick summary of the data.

        Returns:
            DataFrame or str: A summary of the dataset or a message indicating no data.
        """
        if self.data is not None:
            return self.data.describe(include='all')
        return "No data available for analysis."
    
    def get_info(self):
        """
        Display information about the data (e.g., column types, non-null counts).

        Returns:
            str: The information of the dataset or a message indicating no data.
        """
        if self.data is not None:
            buffer = io.StringIO()
            self.data.info(buf=buffer)
            return buffer.getvalue()
        return "No data available."
    
    def get_statistics(self):
        """
        Display basic statistics (e.g., mean, std, min, max) for numeric data.

        Returns:
            DataFrame or str: Statistics of the dataset or a message indicating no data.
        """
        if self.data is not None:
            return self.data.describe()
        return "No data available."
    
    def get_columns(self):
        """
        Get the list of column names.

        Returns:
            list or str: List of column names or a message indicating no data.
        """
        if self.data is not None:
            return self.data.columns.tolist()
        return "No data available."
    
    def get_shape(self):
        """
        Get the shape of the dataset (rows, columns).

        Returns:
            tuple or str: The shape of the dataset or a message indicating no data.
        """
        if self.data is not None:
            return self.data.shape
        return "No data available."
    
    def missing_values(self):
        """
        Show the number of missing values in each column.

        Returns:
            Series or str: The count of missing values per column or a message indicating no data.
        """
        if self.data is not None:
            return self.data.isnull().sum()
        return "No data available for analysis."
    
    def fill_missing(self, strategy='mean'):
        """
        Fill missing values in the data using the specified strategy.

        Args:
            strategy (str): The strategy to use for filling missing values ('mean', 'median', 'mode').

        Returns:
            str: A success or error message.
        """
        if self.data is not None:
            for column in self.data.columns:
                if self.data[column].isnull().sum() > 0:
                    if self.data[column].dtype == np.number:
                        if strategy == 'mean':
                            self.data[column].fillna(self.data[column].mean(), inplace=True)
                        elif strategy == 'median':
                            self.data[column].fillna(self.data[column].median(), inplace=True)
                        elif strategy == 'mode':
                            self.data[column].fillna(self.data[column].mode()[0], inplace=True)
                    else:
                        self.data[column].fillna(self.data[column].mode()[0], inplace=True)
            return "Missing values replaced successfully."
        return "No data available for analysis."
    
    def outlier_detection(self, column):
        """
        Detect outliers in a column using the IQR (Interquartile Range) method.

        Args:
            column (str): The column name to check for outliers.

        Returns:
            DataFrame or str: DataFrame of detected outliers or a message indicating an issue.
        """
        if self.data is not None and column in self.data.columns:
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.data[(self.data[column] < (Q1 - 1.5 * IQR)) | (self.data[column] > (Q3 + 1.5 * IQR))]
            return outliers
        return "Column not found or no data available."
    
    def drop_outliers(self, column):
        """
        Remove outliers from a column using the IQR method.

        Args:
            column (str): The column name from which to remove outliers.

        Returns:
            str: Success or failure message.
        """
        if self.data is not None and column in self.data.columns:
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            self.data = self.data[~((self.data[column] < (Q1 - 1.5 * IQR)) | (self.data[column] > (Q3 + 1.5 * IQR)))]
            return "Outliers removed successfully."
        return "Column not found or no data available."
    
    def drop_duplicates(self):
        """
        Remove duplicate rows from the data.

        Returns:
            str: Success or failure message.
        """
        if self.data is not None:
            self.data.drop_duplicates(inplace=True)
            return "Duplicates removed successfully."
        return "No data available for analysis."
    
    def filter_data(self, condition):
        """
        Filter data based on a condition (query string).
        
        Args:
            condition (str): The condition string to filter the data.
        
        Returns:
            DataFrame or str: Filtered data or a message indicating no data.
        """
        if self.data is not None:
            # Print the columns of the DataFrame for debugging
            print("Columns available:", self.data.columns)
            
            try:
                # Try to filter the data based on the condition
                filtered_data = self.data.query(condition)
                return filtered_data
            except Exception as e:
                return f"Error while filtering data: {e}"
        return "No data available for analysis."

    
    def group_by(self, column):
        """
        Group data by a specific column and return the mean of each group.

        Args:
            column (str): The column to group by.

        Returns:
            DataFrame or str: The grouped data or a message indicating no data.
        """
        if self.data is not None and column in self.data.columns:
            grouped_data = self.data.groupby(column).mean()
            return grouped_data
        return "Column not found or no data available."
    
    def sort_data(self, column, ascending=True):
        """
        Sort data by a specific column.

        Args:
            column (str): The column to sort by.
            ascending (bool): Whether to sort in ascending order. Default is True.

        Returns:
            DataFrame or str: The sorted data or a message indicating no data.
        """
        if self.data is not None and column in self.data.columns:
            sorted_data = self.data.sort_values(by=column, ascending=ascending)
            return sorted_data
        return "Column not found or no data available."
    
    def save_data(self, file_path):
        """
        Save the dataset to a CSV, Excel, or JSON file.

        Args:
            file_path (str): The path to save the file.

        Returns:
            str: Success or failure message.
        """
        ext = os.path.splitext(file_path)[1]
        try:
            if ext == '.csv':
                self.data.to_csv(file_path, index=False)
            elif ext in ['.xls', '.xlsx']:
                self.data.to_excel(file_path, index=False)
            elif ext == '.json':
                self.data.to_json(file_path, orient='records')
            else:
                raise ValueError("File format not supported")
        except Exception as e:
            return f"Error occurred while saving data: {e}"
        return "Data saved successfully."
    def copy(self):
        """
        Create a copy of the data.

        Returns:
            DataFrame or str: A copy of the dataset or a message indicating no data.
        """
        if self.data is not None:
            return self.data.copy()
        return "No data available."
    def Series(self, column):
        """
        Convert a column to a Series object.

        Args:
            column (str): The column name to convert.

        Returns:
            Series or str: The Series object or a message indicating no data.
        """
        if self.data is not None and column in self.data.columns:
            return self.data[column]
        return "Column not found or no data available."


    
    
    
    def plot_correlation(self, save_path=None):
        """
        Plot the correlation matrix of the numerical columns in the data.
        
        Args:
            save_path (str, optional): The path to save the image, if provided.
        """
        if self.data is None:
            print("No data to plot.")
            return
        
        # Select only the numerical columns for correlation
        numeric_data = self.data.select_dtypes(include=[float, int])
        
        # Compute the correlation matrix
        correlation_matrix = numeric_data.corr()
        
        # Create the figure
        plt.figure(figsize=(10, 6))
        
        # Plot the heatmap of the correlation matrix
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        
        # Save the image if the save path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"Correlation plot saved as {save_path}")
        
        # Show the plot
        plt.show()
    
    def save_plot(self, file_path):
        """
        Save the plot to a file
        
        Args:
            file_path (str): The path where the plot image will be saved.
            
        Returns:
            str: Message indicating whether the plot was saved successfully or not.
        """
        if self.data is not None:
            plt.figure(figsize=(10, 6))
            sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title("Correlation Matrix")
            if file_path:
                plt.savefig(file_path, dpi=300)
                plt.close()
                return f"Plot saved successfully at {file_path}."
        return "No data available for plotting."
    
    def show_plot(self, save_path=None):
        """
        Display the plot and save it to a file if save_path is provided
        
        This method shows the correlation matrix plot if data is available, 
        and saves the plot if save_path is provided.
        """
        if self.data is not None:
            plt.figure(figsize=(10, 6))
            sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title("Correlation Matrix")
            
            # Save the plot if save_path is provided
            if save_path:
                plt.savefig(save_path, dpi=300)
                print(f"Plot saved successfully at {save_path}")
            
            plt.show()
        else:
            print("No data available for plotting.")
    
    def plot_pairplot(self, save_path=None):
        """
        Plot relationships between all columns using a pairplot
        
        Args:
            save_path (str, optional): The path to save the image, if provided.
        """
        if self.data is not None:
            pairplot = sns.pairplot(self.data)
            plt.title("Pairplot for all columns")
            
            # Save the plot if save_path is provided
            if save_path:
                pairplot.savefig(save_path)
                print(f"Plot saved successfully at {save_path}")
            
            plt.show()
        else:
            print("No data available for plotting.")
   
   
    def plot_boxplot(self, column, save_path=None):
        """
        Plot outliers using a boxplot
        
        Args:
            column (str): The name of the column to create a boxplot for.
            save_path (str, optional): The path to save the image, if provided.
        """
        if self.data is not None and column in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.data[column])
            plt.title(f"Boxplot of values in column: {column}")
            plt.xlabel(column)
            
            # Save the plot if save_path is provided
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved successfully at {save_path}")
            
            plt.show()
        else:
            print("Column not found or no data available.")
    
    def plot_scatter(self, x_col, y_col, save_path=None):
        """
        Plot the relationship between two columns using a scatter plot
        
        Args:
            x_col (str): The name of the column for the x-axis.
            y_col (str): The name of the column for the y-axis.
            save_path (str, optional): The path to save the image, if provided.
        """
        if self.data is not None and x_col in self.data.columns and y_col in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=self.data[x_col], y=self.data[y_col])
            plt.title(f"Scatter Plot between {x_col} and {y_col}")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            
            # Save the plot if save_path is provided
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved successfully at {save_path}")
            
            plt.show()
        else:
            print("One of the columns not found or no data available.")
    
    def plot_pairplot(self, save_path=None):
        """
        Plot relationships between all columns using a pairplot
        
        Args:
            save_path (str, optional): The path to save the image, if provided.
        """
        if self.data is not None:
            sns.pairplot(self.data)
            plt.title("Pairplot for all columns")
            
            # Save the plot if save_path is provided
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved successfully at {save_path}")
            
            plt.show()
        else:
            print("No data available for plotting.")
    
    def plot_violin(self, column, save_path=None):
        """
        Plot distribution of values in a column using a violin plot
        
        Args:
            column (str): The name of the column to create a violin plot for.
            save_path (str, optional): The path to save the image, if provided.
        """
        if self.data is not None and column in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.violinplot(x=self.data[column])
            plt.title(f"Violin Plot for values in column: {column}")
            plt.xlabel(column)
            
            # Save the plot if save_path is provided
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved successfully at {save_path}")
            
            plt.show()
        else:
            print("Column not found or no data available.")
    
    def plot_heatmap(self, save_path=None):
        """
        Plot correlation matrix using a heatmap
        
        Args:
            save_path (str, optional): The path to save the image, if provided.
        """
        if self.data is not None:
            plt.figure(figsize=(10, 6))
            sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title("Correlation Matrix")
            
            # Save the plot if save_path is provided
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved successfully at {save_path}")
            
            plt.show()
        else:
            print("No data available for plotting.")
    
    def plot_countplot(self, column, save_path=None):
        """
        Plot value counts for a specific column using a count plot
        
        Args:
            column (str): The name of the column to create a count plot for.
            save_path (str, optional): The path to save the image, if provided.
        """
        if self.data is not None and column in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=self.data[column])
            plt.title(f"Countplot for values in column: {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            
            # Save the plot if save_path is provided
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved successfully at {save_path}")
            
            plt.show()
        else:
            print("Column not found or no data available.")
    def plot_distribution(self, column, save_path=None):
        """
        Plot the distribution of a specific column using a histogram
        Args:
            column (str): The name of the column to create a histogram for.
            save_path (str, optional): The path to save the image, if provided.
        """
        if self.data is not None and column in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[column], bins=30, kde=True)
            plt.title(f"Distribution of values in column: {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            
            # Save the plot if save_path is provided
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved successfully at {save_path}")
            
            plt.show()
        else:
            print("Column not found or no data available.")
