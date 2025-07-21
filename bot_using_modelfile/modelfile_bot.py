import requests
import json
import gradio as gr
import sys

url = "http://localhost:11434/api/generate"

headers = {
    "Content-Type": "application/json",
}
history = []

def generate_response(prompt):
    global history
    print(f"\n--- NEW REQUEST ---")
    print(f"User prompt: {prompt}")
    
    history.append(prompt)
    final_prompt = "\n".join(history)
    data = {
        "model": "aura",
        "prompt": final_prompt,
        
    }
    
    try:
        print(f"Sending request to: {url}")
        response = requests.post(url, headers=headers, data=json.dumps(data))
        print(f"Response status: {response.status_code}")
        
        # Print entire response text to console regardless of size
        print("FULL RAW RESPONSE:")
        print(response.text)
        print("END OF RESPONSE")
        
        if response.status_code == 200:
            try:
                # Try parsing as a single JSON object first
                response_json = json.loads(response.text)
                actual_response = response_json.get('response', '')
                print(f"PARSED RESPONSE: {actual_response}")
                
                history.append(actual_response)
                return actual_response
            except json.JSONDecodeError:
                # Fall back to line-by-line parsing if single JSON fails
                print("Falling back to line-by-line parsing")
                response_lines = response.text.strip().split('\n')
                for i, line in enumerate(response_lines):
                    print(f"Line {i}: {line[:50]}...")
                    
                try:
                    last_line = response_lines[-1]
                    final_response = json.loads(last_line)
                    actual_response = final_response.get('response', '')
                    print(f"LINE-PARSED RESPONSE: {actual_response}")
                    
                    history.append(actual_response)
                    return actual_response
                except (json.JSONDecodeError, IndexError, KeyError) as e:
                    print(f"Error parsing response: {str(e)}")
                    return f"Error parsing response: {str(e)}"
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return f"Error: {response.status_code}"
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return f"Exception: {str(e)}"

# Force stdout to flush immediately to ensure prints appear in console
sys.stdout.reconfigure(line_buffering=True)

interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs="text"
)

interface.launch()