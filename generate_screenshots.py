from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://127.0.0.1:8000/")

        # Positive example
        page.fill("#inputText", "Implementing sentiment analysis with pretrained BERT model is super easy, clean, and highly accurate!")
        page.click("#analyzeBtn")
        page.wait_for_selector("#result:not(.hidden)")
        page.screenshot(path="static/images/positive_example.png")

        # Negative example
        page.fill("#inputText", "The overall product quality was extremely poor, buggy, and completely unsatisfactory.")
        page.click("#analyzeBtn")
        page.wait_for_selector("#result:not(.hidden)")
        page.screenshot(path="static/images/negative_example.png")

        browser.close()

if __name__ == "__main__":
    run()
