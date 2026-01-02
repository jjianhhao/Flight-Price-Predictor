import asyncio
import csv
from playwright.async_api import async_playwright
from typing import List, Dict, Optional
import pandas as pd
import datetime
import csv
import os

async def setup_browser():
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=False)
    page = await browser.new_page()
    return p, browser, page


async def extract_flight_element_text(flight, selector: str, aria_label: Optional[str] = None) -> str:
    """Extract text from a flight element using selector and optional aria-label."""
    if aria_label:
        element = await flight.query_selector(f'{selector}[aria-label*="{aria_label}"]')
    else:
        element = await flight.query_selector(selector)
    return await element.inner_text() if element else "N/A"


async def scrape_flight_info(flight) -> Dict[str, str]:
    """Extract all relevant information from a single flight element."""
    departure_time = await extract_flight_element_text(flight, 'span', "Departure time")
    arrival_time =  await extract_flight_element_text(flight, 'span', "Arrival time")
    airline = await extract_flight_element_text(flight, ".sSHqwe")
    duration = await extract_flight_element_text(flight, "div.gvkrdb")
    stops =  await extract_flight_element_text(flight, "div.EfT7Ae span.ogfYpf")
    price =  await extract_flight_element_text(flight, "div.FpEdX span")
    return {
        "Departure Time": departure_time,
        "Arrival Time": arrival_time,
        "Airline Company": airline,
        "Flight Duration": duration,
        "Stops": stops,
        "Price": price,
        "Date Purchase": datetime.datetime.now().strftime("%Y-%m-%d")
    }

def save_to_csv(data: List[Dict[str, str]], filename: str = "flight_data.csv") -> None:
    """Save flight data to a CSV file in the same folder as the script."""
    if not data:
        print("No data to save.")
        return
    headers = list(data[0].keys())
    # Write CSV
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if os.stat(filename).st_size == 0:
            writer.writeheader()
        writer.writerows(data)
    clean_data(filename)

    print(f"Saved CSV at: {os.path.abspath(filename)}")

def clean_data(filename: str = "flight_data.csv") -> None:
    """Clean the CSV data by removing duplicates and sorting."""
    df = pd.read_csv(filename)
    df = df.drop_duplicates(inplace=True)
    df["Price"] = df["Price"].replace("[^0-9]", "", regex=True)
    df["Stops"] = df["Stops"].lstrip(" stops")
    df.to_csv(filename, index=False)
    print(f"Cleaned data saved at: {os.path.abspath(filename)}")


async def scrape_flight_data(one_way_url):
    flight_data = []

    playwright, browser, page = await setup_browser()
    
    try:
        await page.goto(one_way_url)
        
        # Wait for flight data to load
        await page.wait_for_selector(".pIav2d")
        
        # Get all flights and extract their information
        flights = await page.query_selector_all(".pIav2d")
        for flight in flights:
            flight_info = await scrape_flight_info(flight)
            flight_data.append(flight_info)
        
        # Save the extracted data in CSV format
        save_to_csv(flight_data)
            
    finally:
        await browser.close()
        await playwright.stop()

if __name__ == "__main__":
    one_way_url = ("https://www.google.com/travel/flights/search?tfs=CBwQAhopEgoyMDI2LTA3LTIyagwIAhIIL20vMDZ0MnRyDQgCEgkvbS8wM3puY2pAAUgBcAGCAQsI____________AZgBAg")

    asyncio.run(scrape_flight_data(one_way_url))