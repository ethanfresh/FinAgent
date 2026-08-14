from finagent.tools.edgar import edgar_filings
from finagent.tools.executives import executive_profile
from finagent.tools.filing_search import filing_search
from finagent.tools.market_data import fundamental_ratios, price_history
from finagent.tools.news import company_news

ALL_TOOLS = [price_history, fundamental_ratios, edgar_filings, company_news, executive_profile, filing_search]
