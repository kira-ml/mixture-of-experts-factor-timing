from dotenv import load_dotenv
import os
from fredapi import Fred

load_dotenv()

api_key = os.getenv('FRED_API_KEY')

if api_key:
    fred = Fred(api_key=api_key)
    cpi = fred.get_series('CPIAUCSL')
    print(f"✅ FRED is working! CPI data: {cpi.shape}")
else:
    print("❌ No API key found. Check your .env file.")