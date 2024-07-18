import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from textblob import TextBlob

# Load the data
data = pd.read_csv('gpay_transactions.csv')

# Adding a sample transaction description for demonstration
data['description'] = [
    "Bought a new book", "Coffee with friends", "Ride to airport", "Grocery shopping",
    "Monthly subscription", "Morning coffee", "Electronics purchase", "Taxi ride",
    "Music subscription", "Weekly groceries", "Afternoon coffee", "Shopping spree",
    "Airport ride", "Movie night", "Lunch with colleagues", "Online shopping",
    "Commute to work", "Music renewal", "Weekend groceries", "Evening snack"
]

# Sentiment Analysis
data['sentiment'] = data['description'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Data Preprocessing
categorical_features = ['merchant', 'category']
numerical_features = ['amount', 'timestamp', 'sentiment']

# Create preprocessing pipelines for numerical and categorical features
numerical_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine the numerical and categorical pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

# Preprocess the data
data_preprocessed = preprocessor.fit_transform(data)

# Feature Engineering
data['date'] = pd.to_datetime(data['timestamp'], unit='s')
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['hour'] = data['date'].dt.hour

# Aggregate features
data['daily_spending'] = data.groupby(data['date'].dt.date)['amount'].transform('sum')
data['weekly_spending'] = data.groupby(data['date'].dt.isocalendar().week)['amount'].transform('sum')
data['monthly_spending'] = data.groupby(data['date'].dt.month)['amount'].transform('sum')

# Clustering
clustering_features = data[['daily_spending', 'weekly_spending', 'monthly_spending', 'sentiment']]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster'] = kmeans.fit_predict(clustering_features)

# Classification
X = data_preprocessed
y = data['category']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classification model
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# Function to preprocess user input data
def preprocess_user_data(user_data):
    user_data['date'] = pd.to_datetime(user_data['timestamp'], unit='s')
    user_data['day_of_week'] = user_data['date'].dt.dayofweek
    user_data['month'] = user_data['date'].dt.month
    user_data['hour'] = user_data['date'].dt.hour
    
    # Aggregate features
    user_data['daily_spending'] = user_data.groupby(user_data['date'].dt.date)['amount'].transform('sum')
    user_data['weekly_spending'] = user_data.groupby(user_data['date'].dt.isocalendar().week)['amount'].transform('sum')
    user_data['monthly_spending'] = user_data.groupby(user_data['date'].dt.month)['amount'].transform('sum')
    
    # Sentiment analysis for user input
    user_data['sentiment'] = user_data['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    return user_data

# Function to get user input and provide recommendations
def get_user_input_and_recommend():
    # Sample input data
    user_input = {
        'merchant': input("Enter merchant: "),
        'category': input("Enter category: "),
        'amount': float(input("Enter amount: ")),
        'timestamp': int(input("Enter timestamp (in seconds since epoch): ")),
        'description': input("Enter description: ")
    }

    # Convert to DataFrame
    user_data = pd.DataFrame([user_input])
    user_data = preprocess_user_data(user_data)
    
    # Ensure user_data is a DataFrame with a single row
    user_data_for_cluster = user_data[['daily_spending', 'weekly_spending', 'monthly_spending', 'sentiment']]
    
    # Predict the user's cluster
    user_cluster = kmeans.predict(user_data_for_cluster)
    
    # Provide recommendations based on the user's cluster
    if user_cluster == 0:
        recommendation = "You tend to spend a lot on weekends. Consider budgeting for weekend activities."
    elif user_cluster == 1:
        recommendation = "You have high monthly expenses. Try to set aside savings at the beginning of each month."
    elif user_cluster == 2:
        recommendation = "Your transactions show high positive sentiment. Keep track of your spending on items that make you happy."
    elif user_cluster == 3:
        recommendation = "Consider reducing spending on low-sentiment transactions. Look for alternatives that provide better value."
    else:
        recommendation = "General recommendation: Track your expenses regularly to optimize your spending."
    
    print(recommendation)

# Function to generate the full visualization report
def generate_visualization_report(data):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Daily Spending Trends
    axs[0, 0].plot(data['date'], data['daily_spending'], marker='o', linestyle='-', color='b')
    axs[0, 0].set_title('Daily Spending Trends')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Daily Spending ($)')
    axs[0, 0].grid(True)

    # Category Distribution
    sns.countplot(x='category', data=data, palette='viridis', ax=axs[0, 1])
    axs[0, 1].set_title('Transaction Category Distribution')
    axs[0, 1].set_xlabel('Category')
    axs[0, 1].set_ylabel('Count')
    axs[0, 1].tick_params(axis='x', rotation=45)

    # Sentiment Distribution
    sns.histplot(data['sentiment'], bins=20, kde=True, color='g', ax=axs[1, 0])
    axs[1, 0].set_title('Sentiment Analysis of Transactions')
    axs[1, 0].set_xlabel('Sentiment Polarity')
    axs[1, 0].set_ylabel('Count')

    # Clustering Results
    sns.scatterplot(x='daily_spending', y='monthly_spending', hue='cluster', data=data, palette='Set1', legend='full', ax=axs[1, 1])
    axs[1, 1].set_title('Clustering of Spending Behavior')
    axs[1, 1].set_xlabel('Daily Spending')
    axs[1, 1].set_ylabel('Monthly Spending')

    plt.tight_layout()
    plt.show()

# Function to view transaction summary
def view_transaction_summary(data):
    total_spending = data['amount'].sum()
    average_transaction = data['amount'].mean()
    most_frequent_category = data['category'].mode()[0]

    print("\nTransaction Summary:")
    print(f"Total Spending: {total_spending:.2f}")
    print(f"Average Transaction Amount: {average_transaction:.2f}")
    print(f"Most Frequent Category: {most_frequent_category}")

# Function to view detailed analysis by category
def view_category_analysis(data):
    category = input("Enter the category you want to analyze: ")
    category_data = data[data['category'] == category]
    
    if category_data.empty:
        print(f"No transactions found for category: {category}")
        return
    
    total_spending = category_data['amount'].sum()
    average_transaction = category_data['amount'].mean()
    
    print(f"\nDetailed Analysis for Category: {category}")
    print(f"Total Spending: {total_spending:.2f}")
    print(f"Average Transaction Amount: {average_transaction:.2f}")

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Spending Over Time
    axs[0].plot(category_data['date'], category_data['amount'], marker='o', linestyle='-', color='b')
    axs[0].set_title(f'{category} Spending Over Time')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Amount ($)')
    axs[0].grid(True)

    # Sentiment Distribution
    sns.histplot(category_data['sentiment'], bins=20, kde=True, color='g', ax=axs[1])
    axs[1].set_title(f'Sentiment Analysis for {category}')
    axs[1].set_xlabel('Sentiment Polarity')
    axs[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

# Simple chatbot interface
def chatbot_interface():
    print("Welcome to the Transaction Analysis Chatbot!")
    while True:
        print("\nPlease choose an option:")
        print("1. Enter new transaction and get recommendation")
        print("2. Generate visualization report")
        print("3. View transaction summary")
        print("4. View detailed analysis by category")
        print("5. Exit")
        choice = input("Enter your choice (1/2/3/4/5): ")

        if choice == '1':
            get_user_input_and_recommend()
        elif choice == '2':
            generate_visualization_report(data)
        elif choice == '3':
            view_transaction_summary(data)
        elif choice == '4':
            view_category_analysis(data)
        elif choice == '5':
            print("Thank you for using the WalletWizard Chatbot! Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

# Start the chatbot
chatbot_interface()
