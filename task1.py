from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage

sys_template="""Imagine you are running a fictional pizzeria. You need to set up a chat bot that can take orders from the customers based on the
menu you have. Based on the order from customer, you need to sum the total amount and communicate to the customer. If the item is not available,
you need to tell the customer that the 'Item is not available' and ask user to pick from the options you have. You don't need to set up any payment
method, you only need to show the final amount."""

menu="""
    Pepperoni Pizza: $12.95 (large), $10.00 (medium), $7.00 (small)
    Cheese Pizza: $10.95 (large), $9.25 (medium), $6.50 (small)
    Eggplant Pizza: $11.95 (large), $9.75 (medium), $6.75 (small)
    Fries: $4.50 (large), $3.50 (small)
    Greek Salad: $7.25
    Toppings:
        Extra Cheese: $2.00
        Mushrooms: $1.50
        Sausage: $3.00
        Canadian Bacon: $3.50
        AI Sauce: $1.50
        Peppers: $1.00
    Drinks:
        Coke: $3.00 (large), $2.00 (medium), $1.00 (small)
        Sprite: $3.00 (large), $2.00 (medium), $1.00 (small)
        Bottled Water: $5.00"""
        
prompt=ChatPromptTemplate.from_messages([
    ("system",sys_template),
    ("user","{input}")
    
])
from langchain_core.output_parsers import StrOutputParser
output_parser=StrOutputParser()


chat_history=[]

def chat(user):
    chain= prompt | llm | output_parser
    result=chain.invoke({"input":user,"old_chat":chat_history,"menu":menu})
    return result

while True:
    user_ask=input(">>")
    chat_history.append(HumanMessage(content=user_ask))
    result = chat(user_ask)
    chat_history.append(AIMessage(content=result))
    print(">>>",result)
