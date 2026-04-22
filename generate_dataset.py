import os
import json
import time
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

# Setup the Groq client
client = OpenAI(
    api_key="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", 
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"
TOTAL_EMAILS_NEEDED = 500
BATCH_SIZE = 25

class GeneratedEmail(BaseModel):
    id: str = Field(description="A unique ID like 'email_105'")
    sender: str = Field(description="Realistic sender email address")
    subject: str = Field(description="Compelling, realistic subject line")
    body: str = Field(description="Detailed email body with realistic corporate tone, slang, or urgency")
    intent: str = Field(description="MUST be one of the 10 predefined intents")
    priority: int = Field(description="Urgency from 1 to 5")
    is_vip: bool = Field(description="True if sender is C-suite or massive enterprise client")
    department: str = Field(description="Suggested department for human routing (e.g., 'Security', 'HR', 'IT')")

class EmailBatch(BaseModel):
    emails: List[GeneratedEmail]

def generate_batch(batch_num: int, start_id: int) -> List[dict]:
    system_prompt = """You are an expert Data Synthesizer for an AI startup. 
    Your job is to generate a batch of highly diverse, realistic, and sometimes chaotic emails for a corporate inbox.
    
    The 'intent' field MUST be chosen strictly from this list:
    1. routine_password_reset
    2. angry_client_refund
    3. vip_server_outage
    4. general_inquiry
    5. spam
    6. invoice_discrepancy
    7. hr_sensitive
    8. spear_phishing
    9. feature_request
    10. mixed_churn

    Make the emails tricky, lengthy, and highly realistic. 
    CRITICAL INSTRUCTIONS FOR EMAIL BODY:
    - Every email MUST be at least 3 to 4 paragraphs long.
    - Include corporate jargon, confusing tangents, and passive-aggressive tones.
    - Add realistic artifacts like "Sent from my iPhone", massive corporate signatures, confidentiality disclaimers, or fake forwarded message headers.
    - Hide the true intent of the email deep inside the text so the AI has to actually read it to understand it.Include typos, passive-aggressive tones, confusing formatting, and extreme edge cases.
    Make sure the sender domains match the context (e.g., spam from weird domains, VIPs from Fortune 500 domains)."""

    user_prompt = f"Generate {BATCH_SIZE} unique emails. Start the IDs at email_{start_id}."

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "generate_email_batch",
                    "description": "Output the generated emails",
                    "parameters": EmailBatch.model_json_schema()
                }
            }],
            tool_choice={"type": "function", "function": {"name": "generate_email_batch"}}
        )
        
        tool_call = response.choices[0].message.tool_calls[0]
        batch_data = json.loads(tool_call.function.arguments)
        return batch_data.get("emails", [])
        
    except Exception as e:
        print(f"Error generating batch {batch_num}: {e}")
        return []

def main():
    all_emails = []
    print(f"Starting generation of {TOTAL_EMAILS_NEEDED} emails using {MODEL} via Groq...")
    
    num_batches = TOTAL_EMAILS_NEEDED // BATCH_SIZE
    
    for i in range(num_batches):
        start_id = (i * BATCH_SIZE) + 1
        print(f"Generating batch {i+1}/{num_batches} (IDs {start_id} to {start_id + BATCH_SIZE - 1})...")
        
        batch = generate_batch(i+1, start_id)
        if batch:
            all_emails.extend(batch)
            print(f"✅ Successfully added {len(batch)} emails. Total so far: {len(all_emails)}")
        
        time.sleep(2)

    with open("dataset.json", "w") as f:
        json.dump(all_emails, f, indent=4)
        
    print(f"\n🎉 Done! {len(all_emails)} extremely diverse emails saved to dataset.json.")

if __name__ == '__main__':
    main()
