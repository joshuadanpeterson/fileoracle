"""
google_integration.py

This module provides functions to authenticate with the Google Docs API and
to extract text from Google Docs. It uses OAuth2 for authentication.
"""

import os
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these SCOPES, delete the token.json file.
SCOPES = ["https://www.googleapis.com/auth/documents.readonly"]


def authenticate_google_docs():
    """
    Authenticate and create a Google Docs API service instance.

    :return: Google Docs service object.
    """
    creds = None
    if os.path.exists("token.json"):
        from google.oauth2.credentials import Credentials

        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request

            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for next time
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    service = build("docs", "v1", credentials=creds)
    return service


def read_google_doc(doc_id):
    """
    Read and return the content of a Google Doc by its document ID.

    :param doc_id: The Google Doc ID.
    :return: Extracted text content.
    """
    service = authenticate_google_docs()
    document = service.documents().get(documentId=doc_id).execute()
    content = document.get("body").get("content")
    full_text = ""
    for element in content:
        if "paragraph" in element:
            elements = element["paragraph"].get("elements", [])
            for elem in elements:
                text_run = elem.get("textRun")
                if text_run:
                    full_text += text_run.get("content", "")
    return full_text


def read_google_sheet(sheet_id):
    """
    Read and return the content of a Google Sheet by its sheet ID.

    The function uses the Google Sheets API to fetch spreadsheet data,
    then converts the cells into a plain text representation.

    :param sheet_id: The Google Sheet ID.
    :return: Extracted text content from the sheet.
    """
    # Define the scope for reading Google Sheets.
    SCOPES_SHEET = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

    # Authenticate similarly as for Google Docs.
    creds = None
    if os.path.exists("token_sheet.json"):
        from google.oauth2.credentials import Credentials

        creds = Credentials.from_authorized_user_file("token_sheet.json", SCOPES_SHEET)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request

            creds.refresh(Request())
        else:
            from google_auth_oauthlib.flow import InstalledAppFlow

            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES_SHEET
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for next time
        with open("token_sheet.json", "w") as token:
            token.write(creds.to_json())

    from googleapiclient.discovery import build

    service = build("sheets", "v4", credentials=creds)

    # Specify the range you want to retrieve (e.g., the entire sheet)
    range_name = "Sheet1"  # Adjust as needed
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
    values = result.get("values", [])

    if not values:
        return "No data found."

    # Convert the 2D list into a plain text string.
    text_lines = []
    for row in values:
        text_lines.append("\t".join(row))

    return "\n".join(text_lines)
