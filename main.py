import gradio as gr
from llm_req import ask_model, start
import os
import subprocess
import threading

# Schedule application shutdown after a short delay.
def exit_app():
    # Terminate the current process and its child processes.
    def stop():
        subprocess.run(
            ["taskkill", "/PID", str(os.getpid()), "/T", "/F"],
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )

    threading.Timer(0.5, stop).start()
    
# Launch a chat interface with the provided handler.
def interface_run(func):
    print("Run")
    gr.ChatInterface(
    fn=func, 
    ).launch(inbrowser=True)

# Build and launch the main chat interface.
def main():
    with gr.Blocks(fill_height=True) as demo:
        ui = gr.ChatInterface(
            fn=ask_model,
            fill_height=True,
            chatbot=gr.Chatbot(
                elem_id="chat",
                scale=1,
                height=None,
            ),
            textbox=gr.Textbox(
                elem_id="chat-input",
                submit_btn="Send",
            ),
        )

        gr.Button("Exit the program").click(
            fn=exit_app,
            queue=False,
            api_visibility="private",
            js="() => { setTimeout(() => window.close(), 1000); }",
        )
        demo.load(start, outputs=[ui.chatbot, ui.chatbot_state])
        demo.unload(exit_app)  # Add this line.
    demo.launch(
        inbrowser=True,
        css="""
        #chat {
            background: var(--background-fill-secondary) !important;
        }

        #chat .bubble-wrap,
        #chat .panel-wrap {
            background: transparent !important;
        }

        #chat, #chat * {
            font-size: 20px !important;
        }

        #chat-input textarea {
            font-size: 20px !important;
        }
        """,
    )

if __name__ == '__main__':
    main()
