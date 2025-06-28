import streamlit as st
import fin_bot


if 'messages' not in st.session_state:
    st.session_state.messages = []


def reset_conversation():
    st.session_state.messages = []

#Set app title
st.set_page_config(page_title="Financial Bot")

#Set page title
st.title("Financial QnA BOT")

col1, col2 = st.columns([3, 1])
with col2:
    st.button("Reset Chat", on_click=reset_conversation)


# Display chat messages from history
for message in st.session_state.messages:
    content = message["content"]
    
    with st.chat_message(message["role"]):
        if(message["role"] == "user"):
            st.markdown(content)
        elif(message["role"] == "assistant"):
            response = message["metadata"]
            url = None

            if((len(response["context"]) > 0 and 'source' in response["context"][0].metadata)):
                url = response["context"][0].metadata["source"]
                st.markdown(f"{content} \n\n sources - [{url}]({url})")
            else:
                st.markdown(content)

            
            
if prompt := st.chat_input("Please ask your question:"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response  = fin_bot.get_response(prompt, st.session_state.messages)

    url = None

    if((len(response["context"]) > 0 and 'source' in response["context"][0].metadata)):
        url = response["context"][0].metadata["source"]
    
    with st.chat_message("assistant"):
        content = response["answer"]

        if(url != None):
            st.markdown(f"{content} \n\n sources - [{url}]({url})")
        else:
            st.markdown(content)

    st.session_state.messages.append({"role": "assistant", "content": response["answer"], "metadata": response})
    
    