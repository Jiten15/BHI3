import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
#from query import *
import time
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objs as go
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.offline as pyo
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
# from pmdarima.arima import auto_arima
# from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import numpy as np



from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
# export OPENAI_API_KEY="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"


st.set_page_config(page_title="Dashboard",page_icon="🌍",layout="wide")
st.subheader("Dashboard")
st.markdown("##")

theme_plotly = None # None or streamlit

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


df=pd.read_excel('ad.xlsx')

# 'Retailer', 'Retailer ID', 'Invoice Date', 'Region', 'State', 'City',
#        'Product', 'Price per Unit', 'Units Sold', 'Total Sales',
#        'Operating Profit', 'Operating Margin', 'Sales Method', 'Date'


df['year'] = pd.to_datetime(df['Invoice Date']).dt.year
df['month'] = pd.to_datetime(df['Invoice Date']).dt.month
df['day'] = pd.to_datetime(df['Invoice Date']).dt.day

# df3= df[['Total Sales','Price per Unit','Units Sold','Operating Profit','Operating Margin']]




def Assistant():
	import os

	os.environ['OPENAI_API_KEY'] = "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"

	# OPENAI_API_KEY="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"


	user_api_key = "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"

	uploaded_file = st.sidebar.file_uploader("upload", type="csv")

	if uploaded_file :
		with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
		  tmp_file.write(uploaded_file.getvalue())
		  tmp_file_path = tmp_file.name

		loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
		data = loader.load()

		embeddings = OpenAIEmbeddings()
		vectors = FAISS.from_documents(data, embeddings)

		chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo', openai_api_key="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"),
		                                                                retriever=vectors.as_retriever())

		def conversational_chat(query):
		  
		  result = chain({"question": query, "chat_history": st.session_state['history']})
		  st.session_state['history'].append((query, result["answer"]))
		  
		  return result["answer"]

		if 'history' not in st.session_state:
		  st.session_state['history'] = []

		if 'generated' not in st.session_state:
		  st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

		if 'past' not in st.session_state:
		  st.session_state['past'] = ["Hey ! 👋"]
		  
		#container for the chat history
		response_container = st.container()
		#container for the user's text input
		container = st.container()

		with container:
		  with st.form(key='my_form', clear_on_submit=True):
		      
		      user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
		      submit_button = st.form_submit_button(label='Send')
		      
		  if submit_button and user_input:
		      output = conversational_chat(user_input)
		      
		      st.session_state['past'].append(user_input)
		      st.session_state['generated'].append(output)

		if st.session_state['generated']:
		  with response_container:
		      for i in range(len(st.session_state['generated'])):
		          message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
		          message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")







def feature_1():

	df_new = df.groupby('Invoice Date').agg({
		'Price per Unit': 'sum',
    	'Units Sold': 'sum',
    	'Total Sales': 'sum',
    	'Operating Profit': 'sum',
    	'Operating Margin': 'mean'}).reset_index()


	df_new['Invoice Date'] = pd.to_datetime(df_new['Invoice Date'])

	

	selected_column = st.sidebar.selectbox("Select a Feature:", ['Total Sales','Price per Unit','Units Sold','Operating Profit','Operating Margin'])



	st.title("Date Range should be in between Jan,2020-Dec,2021")

	# Sidebar for user input
	start_date = st.sidebar.date_input("Select a Start Date")
	end_date = st.sidebar.date_input("Select an End Date")

	# Check if the start date is before the end date
	if start_date <= end_date:
	  st.write(f"Start Date: {start_date}")
	  st.write(f"End Date: {end_date}")
	  
	  # Convert dates to datetime objects for filtering
	  start_date = datetime.combine(start_date, datetime.min.time())
	  end_date = datetime.combine(end_date, datetime.max.time())
	  

	st.title("Time Period Selector")

	time_period = st.sidebar.selectbox("Select a Time Period:", ["Monthly", "Quarterly", "Yearly","Daily"])



	if time_period == "Monthly":
	  dates = pd.date_range(start=start_date,end=end_date,freq="MS")
	  # monthly_df = df[df['date'].isin(monthly_dates)].copy()
	  # dates_d=monthly_dates
	elif time_period == "Quarterly":
	  dates = pd.date_range(start=start_date,end=end_date,freq="QS")
	  # quarterly_df = df[df['date'].isin(quarterly_dates)].copy()
	  # dates_d=quarterly_dates
	elif time_period == "Yearly":
	  dates = pd.date_range(start=start_date,end=end_date,freq="Y")
	elif time_period == "Daily":
	  dates = pd.date_range(start=start_date,end=end_date,freq="D")



	st.title("Click below button to Plot")

	if st.button("Plot Graph"):

		filtered_df=df_new[df_new['Invoice Date'].isin(dates)]


		def plot(x1,y1):

				
			trace1 = go.Scatter(x=x1,y=y1, mode='lines+markers', name='Actual')
			layout = go.Layout(title="Actual")
			fig = go.Figure(data=[trace1], layout=layout)
			st.plotly_chart(fig)


		plot(filtered_df['Invoice Date'],filtered_df[selected_column])


def feature_2():

	df_new = df.groupby('Invoice Date').agg({
		'Price per Unit': 'sum',
    	'Units Sold': 'sum',
    	'Total Sales': 'sum',
    	'Operating Profit': 'sum',
    	'Operating Margin': 'mean'}).reset_index()


	df_new['Invoice Date'] = pd.to_datetime(df_new['Invoice Date'])

	selected_column = st.sidebar.selectbox("Select a Feature:", ['Total Sales','Price per Unit','Units Sold','Operating Profit','Operating Margin'])

	# Display the selected column from the DataFrame
	# st.write(f"Selected Column: {selected_column}")
	# st.write(df[selected_column])

	############ 1 ###############

	st.title("Date Range should be in between Jan,2020-Dec,2021")

	# Sidebar for user input
	start_date = st.sidebar.date_input("First Duration | Select a Start Date")
	end_date = st.sidebar.date_input("First Duration | Select an End Date")

	# Check if the start date is before the end date
	if start_date <= end_date:
	  st.write(f"Start Date: {start_date}")
	  st.write(f"End Date: {end_date}")
	  
	  # Convert dates to datetime objects for filtering
	  start_date = datetime.combine(start_date, datetime.min.time())
	  end_date = datetime.combine(end_date, datetime.max.time())
	  

	st.title("First Duration | Time Period Selector")

	time_period = st.sidebar.selectbox("First Duration | Select a Time Period:", ["Monthly", "Quarterly", "Yearly","Daily"],key="s0")

	if time_period == "Monthly":
	  	dates = pd.date_range(start=start_date,end=end_date,freq="MS")
	elif time_period == "Quarterly":
	  	dates = pd.date_range(start=start_date,end=end_date,freq="QS")
	elif time_period == "Yearly":
	  	dates = pd.date_range(start=start_date,end=end_date,freq="Y")
	elif time_period == "Daily":
		dates = pd.date_range(start=start_date,end=end_date,freq="D")  
	  

	############# 2 ##################  

	st.title("Date Range should be in between Jan,2020-Dec,2021")

	# Sidebar for user input
	start_date1 = st.sidebar.date_input("Second Duration | Select a Start Date",key="sd1")
	end_date1 = st.sidebar.date_input("Second Duration | Select an End Date",key='sd2')

	# Check if the start date is before the end date
	if start_date1 <= end_date1:
	  st.write(f"Start Date: {start_date1}")
	  st.write(f"End Date: {end_date1}")
	  
	  # Convert dates to datetime objects for filtering
	  start_date1 = datetime.combine(start_date1, datetime.min.time())
	  end_date1 = datetime.combine(end_date1, datetime.max.time())
	  

	st.title("Time Period Selector")

	time_period1 = st.sidebar.selectbox("Second Duration | Select a Time Period:", ["Monthly", "Quarterly", "Yearly","Daily"],key="s1")

	if time_period == "Monthly":
	  	dates1 = pd.date_range(start=start_date1,end=end_date1,freq="MS")
	elif time_period == "Quarterly":
	  	dates1 = pd.date_range(start=start_date1,end=end_date,freq="QS")
	elif time_period == "Yearly":
	  	dates1 = pd.date_range(start=start_date1,end=end_date1,freq="Y")
	elif time_period == "Daily":
		dates1 = pd.date_range(start=start_date1,end=end_date1,freq="D")  
        



    

	st.title("Click below button to Compre Plots")

	# Create a button
	if st.button("Plot Graphs"):

		filtered_df1=df_new[df_new['Invoice Date'].isin(dates)]
		filtered_df2=df_new[df_new['Invoice Date'].isin(dates1)]
	  # Function to plot a simple graph
		def plot(x1,y1):
		   trace1 = go.Scatter(x=x1, y=y1, mode='lines+markers', name='Actual')
		   layout = go.Layout(title="Actual")
		   fig = go.Figure(data=[trace1], layout=layout)
		   # pyo.iplot(fig)
		   st.plotly_chart(fig,use_container_width=True)


		# Call the function
		col1, col2= st.columns(2)

		with col1:
			plot(filtered_df1['Invoice Date'],filtered_df1[selected_column])  
		
		with col2:
			plot(filtered_df2['Invoice Date'],filtered_df2[selected_column])       







def feature_forecast():

	st.title("Total Sales Forecasting")

	df['Date'] = pd.to_datetime(df['Invoice Date'])

	df2= df[['Date', 'Total Sales']]
	df2 = df2.groupby('Date')[['Total Sales']].sum().reset_index()

	df2.set_index('Date', inplace=True)

	model=sm.tsa.statespace.SARIMAX(df2['Total Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
	results=model.fit()
	
	start_date = st.sidebar.date_input("Select a Start Date")
	end_date = st.sidebar.date_input("Select an End Date")

  
	if start_date <= end_date:
	   st.write(f"Start Date: {start_date}")
	   st.write(f"End Date: {end_date}")
	     
	   start_date = pd.to_datetime(start_date)
	   end_date = pd.to_datetime(end_date)

	frequency = st.sidebar.selectbox("Select a Time Period Frequency:", ["Monthly", "Quarterly", "Yearly","Daily"])


	if frequency=="Monthly":
	    frequency='MS'
	elif frequency=="Quarterly":
	    frequency='QS'
	elif frequency=="Yearly":
	    frequency='Y'
	elif frequency=="Daily":
	    frequency='D'    
	    
	if st.button("Plot Forecaste"):

		index_future_dates=pd.date_range(start=start_date,end=end_date,freq=frequency)
		#print(index_future_dates
		C=len(index_future_dates)
		pred=results.predict(start=len(df2),end=len(df2)+C-1,typ='levels').rename('ARIMA Predictions')
		#print(comp_pred)
		pred.index=index_future_dates
		# print(pred)

		fig = go.Figure()

		fig.add_trace(go.Scatter(x=pred.index, y=pred, mode='lines', name='Forecast Sales'))

		# Update the layout of the plot
		fig.update_layout(title='Forecasted Sales ',
		                  xaxis_title='Date',
		                  yaxis_title='Total Sales')

		# Show the plot
		st.plotly_chart(fig)





def feature_predictions():

	def cat_var(a):
		return pd.factorize(a)[0]

	df1=df.copy()

	options_Retailer = ['Foot Locker','Walmart','Sports Direct','West Gear',"Kohl's",'Amazon']
	options_Region = ['Northeast', 'South', 'West', 'Midwest', 'Southeast']
	options_State = ['New York', 'Texas', 'California', 'Illinois', 'Pennsylvania','Nevada', 'Colorado', 'Washington', 'Florida', 'Minnesota','Montana',
	'Tennessee', 'Nebraska', 'Alabama', 'Maine', 'Alaska','Hawaii', 'Wyoming', 'Virginia', 'Michigan', 'Missouri', 'Utah','Oregon', 'Louisiana', 'Idaho',
	'Arizona', 'New Mexico', 'Georgia','South Carolina', 'North Carolina', 'Ohio', 'Kentucky','Mississippi', 'Arkansas', 'Oklahoma', 'Kansas', 'South Dakota',
	'North Dakota', 'Iowa', 'Wisconsin', 'Indiana', 'West Virginia','Maryland', 'Delaware', 'New Jersey', 'Connecticut','Rhode Island',
	'Massachusetts', 'Vermont', 'New Hampshire']
	options_City = ['New York', 'Houston', 'San Francisco', 'Los Angeles', 'Chicago','Dallas', 'Philadelphia', 'Las Vegas', 'Denver', 'Seattle',
	'Miami', 'Minneapolis', 'Billings', 'Knoxville', 'Omaha','Birmingham', 'Portland', 'Anchorage', 'Honolulu', 'Orlando',
	'Albany', 'Cheyenne', 'Richmond', 'Detroit', 'St. Louis','Salt Lake City', 'New Orleans', 'Boise', 'Phoenix', 'Albuquerque',
	'Atlanta', 'Charleston', 'Charlotte', 'Columbus', 'Louisville','Jackson', 'Little Rock', 'Oklahoma City', 'Wichita','Sioux Falls', 'Fargo',
	'Des Moines', 'Milwaukee', 'Indianapolis','Baltimore', 'Wilmington', 'Newark', 'Hartford', 'Providence','Boston', 'Burlington', 'Manchester']
	options_Product = ["Men's Street Footwear", "Men's Athletic Footwear","Women's Street Footwear", "Women's Athletic Footwear",
	"Men's Apparel", "Women's Apparel"]
	options_Method = ['In-store', 'Outlet', 'Online']



	# Display a selectbox widget to allow the user to choose an option
	feature1 = st.selectbox("Retailer", options_Retailer)
	df1.loc[df.index[-1] + 1,"Retailer"] = feature1

	feature2 = st.selectbox("Region", options_Region)
	df1.loc[df.index[-1] + 1,"Region"] = feature2

	feature3 = st.selectbox("State", options_State)
	df1.loc[df.index[-1] + 1,"State"] = feature3

	feature4 = st.selectbox("City", options_City)
	df1.loc[df.index[-1] + 1,"City"] = feature4

	feature5 = st.selectbox("Product", options_Product)
	df1.loc[df.index[-1] + 1,"Product"] = feature5

	feature6 = st.sidebar.number_input("Price per Unit", value=50)
	df1.loc[df.index[-1] + 1,"Price per Unit"] = feature6	

	feature7 = st.sidebar.number_input("Units Sold", value=1300)
	df1.loc[df.index[-1] + 1,"Units Sold"] = feature7

	feature8 = st.sidebar.number_input("Operating Profit", value=300000)
	df1.loc[df.index[-1] + 1,"Operating Profit"] = feature8

	feature9 = st.sidebar.number_input("Operating Margin(%)", value=30)
	df1.loc[df.index[-1] + 1,"Operating Margin"] = feature9

	feature10 = st.selectbox("Sales Method", options_Method)
	df1.loc[df.index[-1] + 1,"Sales Method"] = feature10

	df2=df1.copy()

	df2['Retailer']=pd.factorize(df2.Retailer)[0]
	df2['Region']=pd.factorize(df2.Region)[0]
	df2['State']=pd.factorize(df2.State)[0]
	df2['City']=pd.factorize(df2.City)[0]
	df2['Product']=pd.factorize(df2.Product)[0]
	

	df2.rename(columns={'Sales Method':'Method'},inplace=True)

	df2['Method']=pd.factorize(df2.Method)[0]

	df2 = df2.drop('Retailer ID',axis=1)
	df2 = df2.drop('Invoice Date',axis=1)

	df2['Units Sold'] = df2['Units Sold'].astype(int)
	df2['Total Sales']=df2['Total Sales'].astype(float)
	df2['Operating Profit'] = df2['Operating Profit'].astype(float)
	

	X= df2.values[:,(0,1,2,3,4,5,6,8,9,10)]
	Y= df2.values[:,7]

	# X_test=np.array([X[-1]])

	X_t=X[:-2]
	y_t=Y[:-2]

	X_train, _, y_train, _  = train_test_split(X_t, y_t, test_size = 0.25, random_state = 42)

	lr= LinearRegression()

	lr.fit(X_train,y_train)


	# X_test = np.array([[cat_var(feature1), feature2,feature3, feature4,feature5, feature6,feature7, feature8,feature9,feature10]])
	if st.button("Predict Total Sales"):
		y_pred = lr.predict([np.array(X[-1])])
		st.title(f"The Total Sales will be : {round(abs(y_pred[0]),2)}$")
		# st.write(round(abs(y_pred[0]),2))





def Home():
	col1, col2,col3= st.columns(3)
	with col1:
	    ####### Retailer Counts Bar Chart #########
		retailer_counts = df['Retailer'].value_counts().reset_index()
		retailer_counts.columns = ['Retailer', 'Count']
		retailer_counts = retailer_counts.sort_values(by='Count', ascending=False)
		# st.title("Retailer Counts Bar Chart")
		fig1 = px.bar(retailer_counts, x='Retailer', y='Count', title='Retailer Counts')
		st.plotly_chart(fig1,use_container_width=True)

	with col1:
	    # Group the data by retailer and sum the total sales for each retailer
		retailer_sales = df.groupby('Retailer')['Total Sales'].sum().reset_index()
		# Calculate the total sales of all retailers
		total_sales = retailer_sales['Total Sales'].sum()
		# Calculate the market share of each retailer by dividing their total sales by the total sales of all retailers
		retailer_sales['Market Share'] = retailer_sales['Total Sales'] / total_sales
		# st.title("Market Share Pie Chart")
		fig2 = px.pie(retailer_sales, values='Market Share', names='Retailer', title='Market Share of Retailers')
		st.plotly_chart(fig2,use_container_width=True)

	with col2:
	   product_sales = df.groupby(['Retailer', 'Product'])['Total Sales'].sum().reset_index()
	   fig3 = px.bar(product_sales,x='Retailer',y='Total Sales',color='Product',barmode ='group',title='Total Sales by Product and Retailer',labels={'Total Sales':'Sales'},
			category_orders={"Retailer":["Retailer A","Retailer B","Retailer C"]})
	   st.plotly_chart(fig3,use_container_width=True)

	# Second row
	with col2:
	    # Convert the 'year' column to a datetime format
		df['year'] = pd.to_datetime(df['year'], format='%Y')
		# Group the data by Region and year and calculate the total Sales for each group
		region_sales = df.groupby(['Region', 'year'])['Total Sales'].sum().reset_index()
		# st.title("Total Sales by Region Over Time")
		# Create a line chart using Plotly Express
		fig4 = px.line(region_sales, x='year', y='Total Sales', color='Region',
		              title='Total Sales by Region Over Time')
		# Customize the date formatting on the x-axis
		fig4.update_xaxes(
		    dtick="M1",  # Sets the tick frequency to monthly
		    tickformat="%b-%Y"  # Formats the date as "mm-yyyy"
		)
		# Display the line chart in the Streamlit app
		st.plotly_chart(fig4,use_container_width=True)

	with col3:
	    # Group the data by Sales Method and calculate the total Operating Profit for each group
		profit_by_method = df.groupby('Sales Method')['Operating Profit'].sum().reset_index()
		# st.title("Total Operating Profit by Sales Method")
		# Create a bar chart using Plotly Express
		fig5 = px.bar(profit_by_method, x='Sales Method', y='Operating Profit',title='Total Operating Profit by Sales Method')
		# Display the bar chart in the Streamlit app
		st.plotly_chart(fig5,use_container_width=True)

	with col3:
	    # Group the data by Sales Method and calculate the average Total Sales for each group
		sales_by_method = df.groupby('Sales Method')['Total Sales'].mean().reset_index()
		# st.title("Average Total Sales by Sales Method")
		fig6 = px.bar(sales_by_method, x='Sales Method', y='Total Sales',title='Average Total Sales by Sales Method')
		st.plotly_chart(fig6,use_container_width=True)


def sideBar():

 with st.sidebar:
    selected=option_menu(
        menu_title="Main Menu",
        options=["Home","Predictions","Plot a Feature","Compare Two Durations of a Feature","Forecast a Feature", "Assistant"],
        icons=["house"],
        menu_icon="cast",
        default_index=0
    )
 if selected=="Home":
    
    Home()
    
 if selected=="Plot a Feature":
    st.subheader(f"Page: {selected}")
    feature_1()
    

 if selected=="Compare Two Durations of a Feature":
    st.subheader(f"Page: {selected}")
    feature_2()      

 if selected=="Forecast a Feature":
    st.subheader(f"Page: {selected}")
    feature_forecast()

 if selected=="Predictions":
    st.subheader(f"Page: {selected}")
    feature_predictions()

 if selected=="Assistant":
    st.subheader(f"Page: {selected}")
    Assistant()   

sideBar()

# Home()
# feature_1()
# feature_2()
# feature_3()


#theme
hide_st_style=""" 

<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""

    