"""
Utility functions for Databricks ai_query integration.
Enables querying structured table data using natural language.
"""
import logging
import os
from databricks import sql
from databricks.sdk.core import Config

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CATALOG = "dev_structured"
DEFAULT_SCHEMA = "analytics"
DEFAULT_TABLE = "measureresponses_impairment"
DEFAULT_ENDPOINT = "databricks-gpt-oss-120b"

# Table schema information for context
TABLE_SCHEMA_INFO = """
Table: dev_structured.analytics.measureresponses_impairment
Description: Healthcare text data containing responses about disabilities and impairments.

Key columns:
- koo_chimeasureresponseid: Unique identifier for each response
- koo_clientid: Client identifier
- koo_contactid: Contact identifier  
- koo_responseextended: Free-text healthcare response (main content field)
- koo_appcode: Application code
- koo_description: Description field
- createdon: Timestamp when record was created

This table contains approximately 4.9 million records of healthcare text responses
that may mention disabilities, impairments, or health conditions.
"""


def get_table_info() -> str:
    """Return table schema information for display."""
    return TABLE_SCHEMA_INFO


def _get_databricks_config() -> Config:
    """Get Databricks configuration."""
    return Config()


def _execute_sql_with_user_token(query: str, user_token: str) -> dict:
    """
    Execute a SQL query using the user's access token for authentication.
    This method ensures the query runs with the user's permissions.
    
    Args:
        query: The SQL statement to execute
        user_token: The user's access token from X-Forwarded-Access-Token header
        
    Returns:
        Dictionary containing the query results or error information.
    """
    try:
        cfg = _get_databricks_config()
        warehouse_id = os.getenv('DATABRICKS_WAREHOUSE_ID')
        
        if not warehouse_id:
            return {
                "success": False,
                "error": "DATABRICKS_WAREHOUSE_ID environment variable is not set. Please configure it in app.yaml."
            }
        
        if not user_token:
            return {
                "success": False,
                "error": "User access token not available. Please ensure you are running in a Databricks App environment."
            }
        
        with sql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{warehouse_id}",
            access_token=user_token
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                df = cursor.fetchall_arrow().to_pandas()
                
                # Convert DataFrame to list of dictionaries
                results = df.to_dict('records')
                
                return {
                    "success": True,
                    "results": results,
                    "row_count": len(results)
                }
                
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def _execute_sql_with_service_principal(query: str) -> dict:
    """
    Execute a SQL query using Service Principal credentials.
    Fallback method when user token is not available.
    
    Args:
        query: The SQL statement to execute
        
    Returns:
        Dictionary containing the query results or error information.
    """
    try:
        cfg = _get_databricks_config()
        warehouse_id = os.getenv('DATABRICKS_WAREHOUSE_ID')
        
        if not warehouse_id:
            return {
                "success": False,
                "error": "DATABRICKS_WAREHOUSE_ID environment variable is not set. Please configure it in app.yaml."
            }
        
        with sql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{warehouse_id}",
            credentials_provider=lambda: cfg.authenticate
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                df = cursor.fetchall_arrow().to_pandas()
                
                # Convert DataFrame to list of dictionaries
                results = df.to_dict('records')
                
                return {
                    "success": True,
                    "results": results,
                    "row_count": len(results)
                }
                
    except Exception as e:
        logger.error(f"Error executing SQL with service principal: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def _is_count_question(question: str) -> bool:
    """Check if the question is asking for a count or number of records."""
    question_lower = question.lower()
    count_patterns = [
        'how many', 'count', 'number of', 'total', 'records in',
        'rows in', 'entries in', 'how much'
    ]
    return any(pattern in question_lower for pattern in count_patterns)


def _is_sample_question(question: str) -> bool:
    """Check if the question is asking for sample data or examples."""
    question_lower = question.lower()
    sample_patterns = [
        'show me', 'example', 'sample', 'give me', 'list',
        'what are', 'display', 'fetch'
    ]
    return any(pattern in question_lower for pattern in sample_patterns)


def build_count_query_sql(
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    table: str = DEFAULT_TABLE
) -> str:
    """Build a SQL query to count records in the table."""
    return f"SELECT COUNT(*) AS total_records FROM {catalog}.{schema}.{table}"


def build_sample_query_sql(
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    table: str = DEFAULT_TABLE,
    limit: int = 5
) -> str:
    """Build a SQL query to get sample records from the table."""
    return f"""
    SELECT 
        koo_chimeasureresponseid,
        koo_clientid,
        koo_responseextended,
        koo_appcode,
        createdon
    FROM {catalog}.{schema}.{table}
    WHERE koo_responseextended IS NOT NULL 
      AND TRIM(koo_responseextended) != ''
    LIMIT {limit}
    """.strip()


def build_ai_query_with_data_sql(
    user_question: str,
    data_context: str,
    endpoint: str = DEFAULT_ENDPOINT
) -> str:
    """
    Build a SQL query that uses ai_query with actual data context.
    
    Args:
        user_question: The user's natural language question
        data_context: Actual data from the database to include in the prompt
        endpoint: The model serving endpoint name
        
    Returns:
        SQL query string using ai_query function
    """
    escaped_question = user_question.replace("'", "''")
    escaped_data = data_context.replace("'", "''")
    
    prompt = f"""You are a healthcare data analyst assistant analyzing the measureresponses_impairment table.

Here is the actual data from the database:
{escaped_data}

User question: {escaped_question}

Based on the actual data provided above, please answer the user's question directly and accurately."""

    escaped_prompt = prompt.replace("'", "''")
    
    return f"""
    SELECT ai_query(
        '{endpoint}',
        '{escaped_prompt}'
    ) AS response
    """.strip()


def build_data_analysis_sql(
    user_question: str,
    endpoint: str = DEFAULT_ENDPOINT,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    table: str = DEFAULT_TABLE,
    limit: int = 10
) -> str:
    """
    Build a SQL query that analyzes actual data from the table using ai_query.
    
    This query samples records and uses ai_query to analyze them based on the user's question.
    
    Args:
        user_question: The user's natural language question
        endpoint: The model serving endpoint name
        catalog: The catalog name
        schema: The schema name
        table: The table name
        limit: Maximum number of records to analyze
        
    Returns:
        SQL query string
    """
    escaped_question = user_question.replace("'", "''")
    
    sql = f"""
    SELECT 
        koo_chimeasureresponseid,
        koo_responseextended,
        ai_query(
            '{endpoint}',
            CONCAT(
                'Analyze this healthcare response text and answer the following question: {escaped_question}\\n\\nText: ',
                COALESCE(koo_responseextended, 'No text available')
            )
        ) AS analysis
    FROM {catalog}.{schema}.{table}
    WHERE koo_responseextended IS NOT NULL 
      AND TRIM(koo_responseextended) != ''
    LIMIT {limit}
    """
    
    return sql.strip()


def _execute_sql(query: str, user_token: str = None) -> dict:
    """Execute SQL with user token or service principal fallback."""
    if user_token:
        return _execute_sql_with_user_token(query, user_token)
    else:
        return _execute_sql_with_service_principal(query)


def query_impairment_data(
    user_question: str,
    mode: str = "general",
    endpoint: str = DEFAULT_ENDPOINT,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    table: str = DEFAULT_TABLE,
    user_token: str = None
) -> dict:
    """
    Query the impairment data table using ai_query based on user's natural language question.
    
    This function intelligently routes questions:
    - Count/number questions: Execute actual COUNT(*) SQL and use ai_query to format response
    - Sample/example questions: Fetch real data and use ai_query to explain it
    - Analysis questions: Use ai_query row-by-row on actual data
    - General questions: Fetch sample data context and use ai_query to answer
    
    Args:
        user_question: The user's natural language question
        mode: Query mode - "general" for general questions, "analyze" for data analysis
        endpoint: The model serving endpoint name
        catalog: The catalog name
        schema: The schema name
        table: The table name
        user_token: The user's access token from X-Forwarded-Access-Token header (optional)
        
    Returns:
        Dictionary containing the response or error information
    """
    try:
        # Step 1: Detect question type and fetch relevant data
        if _is_count_question(user_question):
            # For count questions, get the actual count first
            count_query = build_count_query_sql(catalog, schema, table)
            logger.info(f"Executing count query: {count_query}")
            
            count_result = _execute_sql(count_query, user_token)
            if not count_result["success"]:
                return count_result
            
            total_records = count_result["results"][0].get("total_records", 0)
            data_context = f"Total records in {catalog}.{schema}.{table}: {total_records:,}"
            
            # Use ai_query to format a nice response with the actual data
            ai_query = build_ai_query_with_data_sql(user_question, data_context, endpoint)
            logger.info(f"Executing ai_query with count data...")
            
            result = _execute_sql(ai_query, user_token)
            
        elif _is_sample_question(user_question):
            # For sample questions, fetch actual sample data
            sample_query = build_sample_query_sql(catalog, schema, table, limit=5)
            logger.info(f"Executing sample query: {sample_query[:100]}...")
            
            sample_result = _execute_sql(sample_query, user_token)
            if not sample_result["success"]:
                return sample_result
            
            # Format sample data for context
            sample_rows = sample_result["results"]
            data_lines = []
            for i, row in enumerate(sample_rows, 1):
                text_preview = str(row.get("koo_responseextended", ""))[:200]
                data_lines.append(f"Record {i}: {text_preview}...")
            data_context = f"Sample records from {catalog}.{schema}.{table}:\n" + "\n".join(data_lines)
            
            # Use ai_query to explain the sample data
            ai_query = build_ai_query_with_data_sql(user_question, data_context, endpoint)
            logger.info(f"Executing ai_query with sample data...")
            
            result = _execute_sql(ai_query, user_token)
            
        elif mode == "analyze":
            # For analysis mode, use row-by-row ai_query on actual data
            query = build_data_analysis_sql(
                user_question=user_question,
                endpoint=endpoint,
                catalog=catalog,
                schema=schema,
                table=table
            )
            logger.info(f"Executing analysis query: {query[:200]}...")
            
            result = _execute_sql(query, user_token)
            
            if result["success"] and result["results"]:
                # Format analysis results
                analyses = []
                for row in result["results"]:
                    if isinstance(row, dict):
                        analyses.append({
                            "id": row.get("koo_chimeasureresponseid", "N/A"),
                            "text_preview": (str(row.get("koo_responseextended", ""))[:100] + "...") if row.get("koo_responseextended") else "N/A",
                            "analysis": row.get("analysis", "No analysis available")
                        })
                return {
                    "success": True,
                    "response": analyses,
                    "mode": "analyze",
                    "record_count": len(analyses)
                }
        else:
            # For general questions, fetch some sample data for context first
            sample_query = build_sample_query_sql(catalog, schema, table, limit=3)
            sample_result = _execute_sql(sample_query, user_token)
            
            if sample_result["success"] and sample_result["results"]:
                sample_rows = sample_result["results"]
                data_lines = []
                for i, row in enumerate(sample_rows, 1):
                    text_preview = str(row.get("koo_responseextended", ""))[:150]
                    data_lines.append(f"Record {i}: {text_preview}...")
                data_context = f"Sample records from {catalog}.{schema}.{table}:\n" + "\n".join(data_lines)
            else:
                data_context = f"Table: {catalog}.{schema}.{table} (unable to fetch sample data)"
            
            # Use ai_query with data context
            ai_query = build_ai_query_with_data_sql(user_question, data_context, endpoint)
            logger.info(f"Executing ai_query with context...")
            
            result = _execute_sql(ai_query, user_token)
        
        # Step 2: Process and return results
        if result["success"]:
            if result["results"]:
                first_result = result["results"][0]
                response_text = first_result.get("response", str(first_result)) if isinstance(first_result, dict) else str(first_result)
                return {
                    "success": True,
                    "response": response_text,
                    "mode": mode
                }
            else:
                return {
                    "success": True,
                    "response": "Query executed successfully but returned no results.",
                    "mode": mode
                }
        else:
            return {
                "success": False,
                "error": result.get("error", "Unknown error occurred")
            }
            
    except Exception as e:
        logger.error(f"Error in query_impairment_data: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def format_analysis_response(analysis_results: list) -> str:
    """
    Format analysis results into a readable string for display.
    
    Args:
        analysis_results: List of analysis dictionaries
        
    Returns:
        Formatted string for display
    """
    if not analysis_results:
        return "No analysis results available."
    
    formatted_parts = []
    for i, result in enumerate(analysis_results, 1):
        part = f"**Record {i}** (ID: {result.get('id', 'N/A')})\n"
        part += f"*Preview:* {result.get('text_preview', 'N/A')}\n"
        part += f"*Analysis:* {result.get('analysis', 'No analysis')}\n"
        formatted_parts.append(part)
    
    return "\n---\n".join(formatted_parts)
