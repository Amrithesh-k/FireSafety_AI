import streamlit as st
import streamlit as st
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Page configurations
st.set_page_config(
    page_title="Streamlit Navbar Example",
    page_icon=":shark:",
    layout="wide",
)

# Create a navigation bar
nav_option = st.sidebar.selectbox(
    "Navigation",
    ("Home", "Machine Learning model", "chatbot"),
)

# Define the content for each page
if nav_option == "Home":
    st.title("FireSafety 101")
    st.write("Protect yourself from getting roasted")
    st.write("")
    st.header("Introduction")
    st.write("Hey guys since you came to this page we are assuming you dont like to get roasted. And so without any further ado we should learn about what to do when a fire breaks out.")
    st.write("Some basic info you need to know beforehand: ")
    st.write("1. The helpline number for firedepartments in India is 101 and police is 100")
    st.write("2. Never break open the window unless absolutely necessary")
    st.write("Possible situations you might face during a fire")
    st.write("1. If trapped in a room")
    st.write("2. If caught in smoke")
    st.write("3. If you live in a high rise apartment")
    st.write("4. If your clothes catch fire")
    st.write("")
    st.write("")
    st.write("If you are trapped in a room try to seal all the cracks present in the room. Cover the cracks with a wet cloth and close the windows only break them in the worst case possible. Call the firefighters and stay on the call with them until they arrive at your room")
    st.write("")
    st.write("If you are caught in smoke drop down and start crawling tilt your head at an angle of 30 to 35 degree then try to breathe shalowly and use your shirt or tshirt as a make do filter. Try to get away from the area with smoke as fast as possible")
    st.write("")
    st.write("If you live in a high rise apartment use a bright colored blanket as symbol to alert the fire department or use torch if the fire happened during night and follow the instruction as stated above")
    st.write("")
    st.write("If your clothes catch fire cover your mouth and face with your hands to prevent burns and then roll on the floor to extinguish the fire")
    st.write("")
    st.header("Instructions to operate a Fire Extinguisher")
    st.write("")
    st.write("**P** - PULL safety pin from handle.")
    st.write("**A** - AIM (nozzle, cone, horn) at base of the fire.")
    st.write("**S** - SQUEEZE the trigger handle.")
    st.write("**S** - SWEEP from side to side (watch for re-flash).")
    st.write("")
    st.write("")
    st.write("For more information you can browse these links: ")
    st.write("[Source material for this document](https://emergency.vt.edu/ready/guides/building-fire/building-fire-during.html)")
    st.write("Statistics for fire accidents and detailed information can be found: [here](https://nidm.gov.in/PDF/pubs/Fires_in_India_2020.pdf)")


elif nav_option == "Machine Learning model":
    st.title("Ai System")
    st.write("Content for Page 1 goes here.")
'''

elif nav_option == "chatbot":
    chatbot = ChatBot("FireSafetyBot")

    # Create a new trainer for the chatbot
    trainer = ChatterBotCorpusTrainer(chatbot)

    # Train the chatbot on the English language corpus data
    trainer.train("chatterbot.corpus.english")

    # Streamlit app
    def main():
        st.title("Fire Safety Chatbot")

        st.sidebar.title("About")
        st.sidebar.info(
            "This is a simple chatbot for fire safety. Ask questions related to fire safety, and the chatbot will provide information."
        )

        user_input = st.text_input("You: ")

        if st.button("Ask"):
            bot_response = chatbot.get_response(user_input)
            st.text_area("FireSafetyBot:", bot_response.text)

        if __name__ == "__main__":
            main()
    
'''

# Add a footer
st.markdown("---")

