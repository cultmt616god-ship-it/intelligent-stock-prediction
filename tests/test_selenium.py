from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

try:
    print("Installing driver...")
    path = ChromeDriverManager().install()
    print(f"Driver installed at: {path}")
    
    options = Options()
    options.add_argument('--headless')
    
    print("Starting service...")
    service = Service(path)
    driver = webdriver.Chrome(service=service, options=options)
    
    print("Getting page...")
    driver.get("https://www.google.com")
    print(driver.title)
    driver.quit()
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
