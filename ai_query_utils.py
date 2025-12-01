"""
Utility functions for Databricks ai_query integration.
Enables querying structured table data using natural language.

Supports two authorization models:
1. User authorization: Uses the user's access token from x-forwarded-access-token header
   to query data on behalf of the user (respects Unity Catalog permissions)
2. App authorization: Uses the app's service principal credentials (fallback for local dev)
"""
import logging
import os
from typing import Optional
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.service.sql import StatementState

logger = logging.getLogger(__name__)


def get_user_access_token() -> Optional[str]:
    """
    Get the user's access token from Databricks Apps HTTP headers.
    
    When running as a Databricks App with user authorization enabled,
    the user's access token is forwarded via the x-forwarded-access-token header.
    
    Returns:
        The user's access token if available, None otherwise.
    """
    try:
        import streamlit as st
        return st.context.headers.get("x-forwarded-access-token")
    except Exception as e:
        logger.debug(f"Could not get user access token from headers: {e}")
        return None


def get_sql_warehouse_http_path() -> Optional[str]:
    """
    Get the SQL warehouse HTTP path from environment variable.
    
    The SQL_WAREHOUSE_HTTP_PATH environment variable is set via app.yaml
    using the valueFrom directive to map the sql-warehouse resource.
    
    Returns:
        The SQL warehouse HTTP path if configured, None otherwise.
    """
    return os.environ.get("SQL_WAREHOUSE_HTTP_PATH")


def _get_workspace_client_with_user_token(user_token: str) -> WorkspaceClient:
    """
    Create a WorkspaceClient that uses the user's access token.
    
    This enables user authorization where the app acts on behalf of the user,
    respecting their Unity Catalog permissions.
    
    Args:
        user_token: The user's access token from x-forwarded-access-token header.
        
    Returns:
        WorkspaceClient configured with user's token.
    """
    host = os.environ.get("DATABRICKS_HOST")
    if not host:
        raise RuntimeError("DATABRICKS_HOST environment variable not set")
    
    cfg = Config(
        host=host,
        token=user_token,
        credentials_strategy="pat",
    )
    return WorkspaceClient(config=cfg)


# Default configuration
DEFAULT_CATALOG = "dev_structured"
DEFAULT_SCHEMA = "analytics"
DEFAULT_TABLE = "measuresponses_impairment"
DEFAULT_ENDPOINT = "databricks-gpt-oss-120b"

# Table schema information for context
TABLE_SCHEMA_INFO = """
Table: dev_structured.analytics.measuresponses_impairment
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


def _get_workspace_client() -> WorkspaceClient:
    """Get a Databricks WorkspaceClient instance."""
    return WorkspaceClient()


def _execute_sql(sql: str, warehouse_id: str = None, use_user_auth: bool = True) -> dict:
    """
    Execute a SQL statement using Databricks SQL Statement Execution API.
    
    Supports two authorization models:
    1. User authorization (default): Uses the user's access token from HTTP headers
       to execute queries on behalf of the user, respecting their Unity Catalog permissions.
    2. App authorization: Uses the app's service principal credentials (fallback).
    
    Args:
        sql: The SQL statement to execute
        warehouse_id: Optional warehouse ID. If not provided, uses default.
        use_user_auth: If True, attempt to use user authorization first (default: True)
        
    Returns:
        Dictionary containing the query results or error information.
    """
    try:
        # Try to use user authorization if enabled
        user_token = None
        if use_user_auth:
            user_token = get_user_access_token()
            if user_token:
                logger.info("Using user authorization for SQL execution")
                w = _get_workspace_client_with_user_token(user_token)
            else:
                logger.info("User token not available, falling back to app authorization")
                w = _get_workspace_client()
        else:
            w = _get_workspace_client()
        
        # If no warehouse_id provided, try to get one from available warehouses
        if not warehouse_id:
            warehouses = list(w.warehouses.list())
            if not warehouses:
                return {
                    "success": False,
                    "error": "No SQL warehouses available. Please configure a SQL warehouse."
                }
            # Use the first available running warehouse, or the first one if none running
            running_warehouses = [wh for wh in warehouses if wh.state and wh.state.value == "RUNNING"]
            warehouse = running_warehouses[0] if running_warehouses else warehouses[0]
            warehouse_id = warehouse.id
            
        # Execute the SQL statement
        response = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=sql,
            wait_timeout="30s"
        )
        
        # Check execution status
        if response.status and response.status.state == StatementState.SUCCEEDED:
            # Extract results
            results = []
            if response.result and response.result.data_array:
                columns = []
                if response.manifest and response.manifest.schema and response.manifest.schema.columns:
                    columns = [col.name for col in response.manifest.schema.columns]
                
                for row in response.result.data_array:
                    if columns:
                        results.append(dict(zip(columns, row)))
                    else:
                        results.append(row)
            
            return {
                "success": True,
                "results": results,
                "row_count": len(results)
            }
        elif response.status and response.status.state == StatementState.FAILED:
            error_msg = "Query execution failed"
            if response.status.error:
                error_msg = response.status.error.message or error_msg
            return {
                "success": False,
                "error": error_msg
            }
        else:
            return {
                "success": False,
                "error": f"Query in unexpected state: {response.status.state if response.status else 'unknown'}"
            }
            
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def build_ai_query_sql(
    user_question: str,
    endpoint: str = DEFAULT_ENDPOINT,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    table: str = DEFAULT_TABLE,
    sample_size: int = 5
) -> str:
    """
    Build a SQL query that uses ai_query to answer questions about the impairment data.
    
    Args:
        user_question: The user's natural language question
        endpoint: The model serving endpoint name
        catalog: The catalog name
        schema: The schema name
        table: The table name
        sample_size: Number of sample records to include for context
        
    Returns:
        SQL query string using ai_query function
    """
    # Escape single quotes in the user question
    escaped_question = user_question.replace("'", "''")
    
    # Build the prompt that includes table context
    prompt = f"""You are a healthcare data analyst assistant. You have access to the {catalog}.{schema}.{table} table which contains healthcare text responses about disabilities and impairments.

Table columns:
- koo_chimeasureresponseid: Unique response ID
- koo_clientid: Client ID
- koo_contactid: Contact ID
- koo_responseextended: Free-text healthcare response content
- koo_appcode: Application code
- koo_description: Description
- createdon: Creation timestamp

The user is asking about this healthcare impairment data. Here is their question:
{escaped_question}

Please provide a helpful, accurate response based on your understanding of healthcare data and disability/impairment terminology. If the question requires specific data analysis, explain what kind of query or analysis would be needed."""

    # Escape the prompt for SQL
    escaped_prompt = prompt.replace("'", "''")
    
    sql = f"""
    SELECT ai_query(
        '{endpoint}',
        '{escaped_prompt}'
    ) AS response
    """
    
    return sql.strip()


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


def query_impairment_data(
    user_question: str,
    mode: str = "general",
    endpoint: str = DEFAULT_ENDPOINT,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    table: str = DEFAULT_TABLE,
    warehouse_id: str = None
) -> dict:
    """
    Query the impairment data table using ai_query based on user's natural language question.
    
    Args:
        user_question: The user's natural language question
        mode: Query mode - "general" for general questions, "analyze" for data analysis
        endpoint: The model serving endpoint name
        catalog: The catalog name
        schema: The schema name
        table: The table name
        warehouse_id: Optional SQL warehouse ID
        
    Returns:
        Dictionary containing the response or error information
    """
    try:
        if mode == "analyze":
            sql = build_data_analysis_sql(
                user_question=user_question,
                endpoint=endpoint,
                catalog=catalog,
                schema=schema,
                table=table
            )
        else:
            sql = build_ai_query_sql(
                user_question=user_question,
                endpoint=endpoint,
                catalog=catalog,
                schema=schema,
                table=table
            )
        
        logger.info(f"Executing ai_query SQL: {sql[:200]}...")
        
        result = _execute_sql(sql, warehouse_id)
        
        if result["success"]:
            # Extract the response from results
            if result["results"]:
                if mode == "analyze":
                    # For analysis mode, format multiple results
                    analyses = []
                    for row in result["results"]:
                        if isinstance(row, dict):
                            analyses.append({
                                "id": row.get("koo_chimeasureresponseid", "N/A"),
                                "text_preview": (row.get("koo_responseextended", "")[:100] + "...") if row.get("koo_responseextended") else "N/A",
                                "analysis": row.get("analysis", "No analysis available")
                            })
                    return {
                        "success": True,
                        "response": analyses,
                        "mode": "analyze",
                        "record_count": len(analyses)
                    }
                else:
                    # For general mode, return the single response
                    first_result = result["results"][0]
                    response_text = first_result.get("response", str(first_result)) if isinstance(first_result, dict) else str(first_result)
                    return {
                        "success": True,
                        "response": response_text,
                        "mode": "general"
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
