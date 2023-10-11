# import gradio as gr

# def print_text(text):
#     print(text)

# if __name__ == "__main__":
#     examples = ['Hello', 'Hi', 'How are you']
#     demo = gr.Interface(print_text, 
#                         inputs='text', 
#                         outputs ='text', 
#                         examples=examples,
#                         cache_examples=False,
#                         description="print_test",
#                         )
    
#     demo.launch(share=True,server_port=7860)

import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
    
if __name__ == "__main__":
    demo.launch(share=True)  