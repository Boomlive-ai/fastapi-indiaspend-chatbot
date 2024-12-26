from typing import Optional
from langchain.schema.runnable import Runnable
from utils import store_articles_custom_range, store_daily_articles

class StoreDailyArticles(Runnable):
    """
    A class to fetch and store articles for the current day.
    """
    async def invoke(self, *args, **kwargs):
        """
        Fetch and store articles for the current day asynchronously.

        Returns:
            dict: A dictionary containing the status and details of the operation.
        """
        try:
            # Fetch and store articles for today
            daily_articles = await store_daily_articles()
            
            # Return success response
            return {
                "status": "success",
                "message": "Articles for today have been successfully stored.",
                "details": daily_articles,
            }
        except Exception as e:
            # Handle any errors
            return {
                "status": "error",
                "message": "Failed to store daily articles.",
                "error": str(e),
            }






class StoreCustomRangeArticles(Runnable):
    """
    A class to fetch and store articles within a custom date range.
    """
    async def invoke(self, from_date: Optional[str] = None, to_date: Optional[str] = None, *args, **kwargs):
        """
        Fetch and store articles within a custom date range asynchronously.

        Args:
            from_date (str, optional): Start date in 'YYYY-MM-DD' format.
            to_date (str, optional): End date in 'YYYY-MM-DD' format.

        Returns:
            dict: A dictionary containing the status and details of the operation.
        """
        try:
            # Call the function to store articles with the custom range
            custom_range_articles = await store_articles_custom_range(from_date, to_date)
            
            # If the operation is successful, return a success message
            return {
                "status": "success",
                "message": f"Articles from {from_date or '6 months ago'} to {to_date or 'today'} have been successfully stored.",
                "details": custom_range_articles,
            }
        except Exception as e:
            # Handle and log any errors that occur during the process
            return {
                "status": "error",
                "message": "Failed to store articles.",
                "error": str(e),
            }

