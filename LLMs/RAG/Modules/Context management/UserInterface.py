import streamlit as st
import requests

# FastAPI URL
API_URL = "http://127.0.0.1:8000/process_query/"

def get_user_input():
    user_name = st.text_input("Enter your name")
    query = st.text_area("Enter your query")
    return user_name, query

def call_api(user_name, query):
    payload = {
        "user_name": user_name,
        "query": query
    }
    response = requests.post(API_URL, json=payload)
    return response.json()

def display_response(response):
    if response:
        st.write("Response from FastAPI:")

def display_response(response):
    if response:
        st.write("Response from FastAPI:")
        st.json(response)

def main():
    st.title("User_management")
    
    user_name, query = get_user_input()
    
    if st.button("Submit"):
        if user_name and query:
            response = call_api(user_name, query)
            display_response(response)
        else:
            st.error("Please provide both user name and query.")

if __name__ == "__main__":
    main()
