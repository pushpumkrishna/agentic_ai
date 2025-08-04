PROMPT_TEMPLATE1 = """
        You are an AI email classifier. Email consist of details or concerns about any purchase order.
        First detect the language of the input. Read the content carefully before following instructions.

        Based on the following JSON layout, classify it into one of the categories:

        {subject}: Subject of the email
        {body}: Content/body of email

        The categories are:
        - Duplicate Invoice
        - 
        - Reminder
        - Shipping Charges
        - Payment Reminder
        - Incorrect Address
        - Invoice
        - Payment Acknowledgement

        Output:
        - Return the response as in below JSON format only. Do not add anything else including comments/messages.

        {{"response": {{"category": <LLM Response>}}}}
        """

PROMPT_TEMPLATE = """
You are an AI email classifier. Email consist of details or concerns about any purchase order.
- First detect the language of the input. Read the content carefully before following instructions.
- Input contains email subject and body. Read it carefully and classify it into one of the below categories: 

- Below is input subject and body: 

{subject}: Subject of the email
{body}: Content/body of email

Following are Categories. Classify based on given information:

1. Statement

1.1. Content Type: A detailed financial document summarizing account activities, often issued periodically 
(e.g., monthly or yearly) to provide an overview of transactions, balances, and other key financial metrics. 
These documents are typically intended to give account holders a clear view of their financial standing, 
including amounts due and outstanding balances.

1.2. Key Indicators:

- Look for keywords and phrases such as "statement," "balance," "account activity," "account summary," "amount due," 
"debit," "credit due," and "outstanding balance."
- Sections showing aggregated figures, such as monthly or yearly totals, balances, or summaries of debits and credits.
- Structured presentation of financial data, with line items detailing individual transactions, payment due dates, and 
account totals to reflect a holistic view of the account's financial status over the stated period.

2. Credit Note 
(with PO)

2.1. Content Type: A financial document issued in association with a purchase order (PO) that provides details of any 
credit issued, often reflecting adjustments, refunds, or corrections related to previous transactions. 
Credit notes are typically used to adjust or offset charges in cases of returns, over-billing, or service adjustments, and provide a clear record for both parties.

2.2. Key Indicators:

- Look for terms such as "Credit Note," "Credit," or phrases indicating credit issuance, often in conjunction with references to "Purchase Order" or "PO number."
- Sections listing credit amounts, detailed purchase information, refund amounts, or specific adjustments made to the original transaction.
- Structured presentation of transaction adjustments, often listing affected line items, original charges, and associated credit amounts to reflect changes tied to purchase orders.

3. Invoice 

(Debit Note / Invoice / Tax Invoice Non-PO)

3.1 Content Type:
- Purpose: This document is an accounting tool used for adjusting or documenting sales or transactions.
- Types: Debit Note, Invoice, or Tax Invoice (Non-PO).
- Usage: These are issued when there are changes to an earlier transaction (like adding charges, correcting errors, or adjusting terms).
- No Purchase Order: Unlike regular invoices that have a linked purchase order (PO), these documents don't require a PO, making them relevant for situations where no formal PO exists.

3.2 Key Indicators & Invoice Queries:
- Terms Used: Look for terms like "Debit Note," "Debit Memo," "Invoice," "Tax Invoice," or "Sales Invoice."
- Amount: Usually associated with a positive total amount (indicating charges or liabilities).
- Vendor and Customer Information: Contains the vendor's name and address, as well as the customer’s (ship-to/sold-to) name and address.
- Reference Number: Often includes a reference number or other identifiers.
- Transaction Details: This may include:
    -Document date
    -Amount
    -Currency
    -Banking details
    -Description of goods/services sold
- Tax and Payment Info: May include details on tax (e.g., VAT or sales tax) and payment instructions, depending on the type of transaction.
- Missing Invoice: Missing invoice or credit/debit note to be requested from vendor. Information in email for the above is missing.
- Return Invoice: Request for returning the wrong invoices due to incorrect amount, PO, VAT, etc.
- Status: To answer general queries regarding the status of an invoice.
- Invoice In Progress: To resolve any issues with the processing or approval of an invoice.
- Workflow Issue: To resolve workflow issues (e.g., requester of PO, owner of CC or WBS has left the company or is absent, looking for additional information).
- Factoring/SCF Supply Chain Finance: Factoring documents and information related to supply chain finance.
- Credit/Debit Note Query: To answer queries related to credit or debit notes.

4. Non PO Invoice 
(Credit Note without PO)

4.1. Content Type: A credit note non-PO is an accounting document used to process returns or adjustments related to 
sales transactions. It serves as proof of the return of goods, lack of service delivery, overcharges, or other 
transaction discrepancies. The document indicates the vendor's payment liability and reflects the necessary 
corrections to the original sale. Unlike standard credit notes, this document does not have a purchase order (PO) 
attached to it, making it distinct in situations where no formal purchase order exists.

4.2. Key Indicators:

- The document contains the wording "Credit Note" or "Credit Memo."
- It usually displays a negative total amount, indicating a refund or price adjustment.
- Mentions of the vendor’s name and address, along with ship-to/sold-to name and address.
- Includes a reference number, date of issuance, and details of the amount, currency, and reason for the return or 
adjustment.
- A description of the goods or services being returned or adjusted, along with the price change or other corrections 
made to the original transaction.

5. Credit Invoice

5.1. Content Type: A credit invoice is an accounting document issued by a seller or vendor to apply an adjustment, 
refund, or discount to a previous transaction or invoice. It is typically used when a customer is entitled to a 
reduction in the amount owed, such as for returns, pricing errors, or other adjustments. 
The credit invoice serves as a formal acknowledgment of the credit applied and reduces the customer’s payment liability. 
Unlike a standard invoice, it may be issued after the original invoice has been paid or partially paid, and it 
effectively lowers the total outstanding amount.

5.2. Key Indicators:

- The document contains the terms "Credit Invoice" or "Credit Memo."
- Usually associated with a positive total amount, but marked as a credit or adjustment to the original transaction.
- Includes the original invoice number, reference to the goods/services involved, and the amount being credited.
- Details of the customer’s name and address, along with vendor information.
- Typically shows the reason for the adjustment, such as a discount, return, or price correction.
- Often includes payment instructions or account details for applying the credit.

6. Payment Reminder
6.1 Content Type: A Payment Reminder is a formal notification or alert, 
typically sent by a business or service provider to a customer, regarding an outstanding or upcoming payment. 
This reminder serves to inform the recipient about a payment that is either overdue or approaching its due date. 
It may be part of a series of reminders that escalate in urgency as the due date nears or passes. 
Payment reminders are used in various industries (e.g., utilities, subscription services, loan repayments) to:

- Ensure timely payments
- Prevent disruptions in service
- Avoid legal consequences for non-payment

Look for these key signs:

- The email is related to timely payment or the prevention of payment delays.
- The reminder may be sent through one of the following channels:
    - Email
    - Text message
    - Physical mail (if mentioned as received by post, and relevant to the payment)
- Additionally, the email may reference various types of documents related to the payment:

- Red Letter Reminders: Identify if the email mentions a Red Letter Reminder, which is an urgent notice often sent via 
    physical mail to highlight a serious overdue payment. This type of reminder may warn of penalties, service 
    disruption, or legal action if payment isn't made immediately.
- Statements/Balance Confirmations: Emails that include a request for balance confirmation or an overview of an 
    outstanding balance. These are typically not invoices but may be part of a routine statement or request for 
    confirmation from the recipient about their balance.
- Balance Reconciliation Requests: Emails asking the recipient to verify or reconcile their balance due, possibly due 
    to discrepancies or ongoing account management.
- Scanning AP/Other Documents: Look for mentions of scanned documents related to accounts payable or other business 
    processes, though these would not include invoices.

6.2 Key Indicators:
To accurately categorize the email as a Payment Reminder, look for the following indicators:

- Urgent Language: The email should contain language suggesting the payment is overdue or needs urgent attention. 
Keywords may include:
    - "Payment Reminder"
    - "Overdue payment"
    - "Please pay by [specific date]"
    - "Past due"
    - "Final notice"
    - "Immediate action required" (for Red Letter Reminders)
- Payment Details: Check if the email includes the following details:
    - Amount due
    - Original invoice number (if applicable)
    - Due date
    - Outstanding balance (for statements or confirmations)
- Payment Instructions: Look for instructions on how the recipient should make the payment. This may include:
    - Payment methods
    - Bank details or a payment portal link
- Request for Immediate Action: The email may request the recipient to settle the payment promptly to avoid penalties, 
    interest, service interruptions, or legal action.
- Associated Documents: The email may also mention related documents such as:
    - Balance reconciliation requests
    - Scanning of AP or other non-invoice documents (not for invoices themselves but for record-keeping or verification)

Summary:
Classify the email as a Payment Reminder if it contains urgent language about overdue or upcoming payments. 
Pay special attention to the use of terms like "final notice," "past due," or "immediate payment required." 
Look for additional references to Red Letter Reminders, statements, balance confirmations, or scanned documents 
related to the accounts payable process. Emails that include these elements, or request payment to avoid penalties, 
service disruption, or legal issues, should be categorized accordingly.

7. Order Confirmation

7.1. Content Type: A document confirming that an order has been successfully placed, providing details about the products or services to be delivered. This document serves as an acknowledgment from the seller to the buyer, verifying receipt of the order and outlining what was purchased.

7.2. Key Indicators:

- Look for phrases such as "order confirmation," "order number," "thank you for your order," or other language indicating confirmation of an order rather than a request for payment.
- Includes a list of items ordered, quantities, product descriptions, and expected delivery dates, giving the buyer a clear summary of the purchase.
- Typically contains shipping and billing addresses, indicating where the order will be delivered, but lacks any mention of payment requirements or due dates, as this document is purely informational rather than a billing request.


8. Reminder

8.1. Content Type: A general notification intended to remind the recipient about a specific task, appointment, event, or action that needs attention. These reminders are typically used for non-payment-related matters and serve as a gentle prompt to ensure important activities are not forgotten. They can be related to meetings, deadlines, follow-up actions, or upcoming events. This type of reminder is often less urgent than a payment reminder and focuses on ensuring tasks or commitments are fulfilled on time.

8.2. Key Indicators:

- Language such as "reminder," "follow-up," "this is a reminder for," or "just a quick reminder" to indicate the purpose of the message.
- References to specific tasks, appointments, events, or actions that require attention, often without any payment-related details.
- Short and concise in nature, typically providing clear instructions or key details.
- May include specific deadlines, important dates, or next steps that the recipient should take in response.


9. Unable to classify
- If no category is decided.


Output:
- Return the response as in below JSON format only. Do not add anything else including comments/messages.
{{"response": {{"category": <LLM Response>}}}}
"""

PROMPT_TEMPLATE_LANGUAGE_GUESS = """
You are an AI language expert. Your task is to identify the language of the given email content. The email consists 
of details or concerns about a purchase order. Analyze the content carefully before guessing the language.

Based on the following JSON layout, guess the language of the subject and body of the email:

{subject}: Subject of email
{body}: Content/body of email

Output:
- Return the response in the following JSON format. Do not add anything else including comments/messages.

{{"response": {{"language": "<Guessed Language>"}}}}
"""


PROMPT_TEMPLATE_INTENT_GUESS = """
You are an AI Email Intent Detection expert. Your task is to identify the detailed intent or purpose of the given email 
content, which may contain details, concerns, or inquiries about a purchase order or related topics. 
Carefully analyze the content to determine the main purpose of the email.

Based on the following JSON layout, identify the intent from both the subject and body of the email:

{subject}: Subject of the email
{body}: Content/body of the email

Instructions:
- Make sure the intent description captures the email’s main purpose with enough detail for clarity.
- In the "suggested_steps" section, provide clear and actionable recommendations for addressing the email’s content.
- Avoid adding any additional comments or messages outside the JSON format.

Output:
- Return the response in the following JSON format with:
  - A detailed description of the intent, outlining the purpose or issue raised in the email. What is being conveyed 
  in the text? What sender is expecting? Make sure it is at least 2-3 lines.

{{"response": {{"intent": "<Detailed Description of Intent in bullet points>"}}}}

"""


# PROMPT_TEMPLATE_EXTRACT_INVOICE_NUMBER = """
# You are an AI information extractor.
# - Your task is to extract the Invoice & E-Invoice Number from the given email subject and body.
# - The email consists of details or concerns about a purchase order and may include various fields such as PO number,
#     invoice number, E-invoice number, vendor number, and the date etc.
# - You are tasked to extract the Invoice Number and E-Invoice Number only.
# - ### Key Instruction: Analyze the content carefully before extracting anything.
#
# Below is the the subject and body of the email:
# {subject}: Subject of email
# {body}: Content/body of email
#
# - **Rules to Identify Invoice Number:**
#     There is no specific pattern for this number. Different vendors use different formats. Still you can follow below
#     rules to extract the Invoice number.
#
#     1. Patterns:
#        - Can start with any sequence of letters or numbers.
#        - ### Key Instruction: Can be purely numeric (e.g., '288617') or alphanumeric (e.g., 'TP0000330').
#        - ### Key Instruction: Mostly Start with "I", "IE" or "INV" and will have numbers after it like "I3453423",
#         "IE87271", "INV1121342112". These examples are just for reference. Do not use it as output.
#        - ### Key Instruction: Can have "INV" or "IN" in middle of the number also.
#        - May include a slash ("/"), hyphen ("-"), Underscore (_) or dot(.).
#        - No other special symbols or spaces are allowed.
#     2. ### Key Instruction: Exclude the number/numbers that follow below rules:
#         - ### Key Instruction: **Never starts with 40, 45, 41, 43, 65, 800, 2000, 3000, 400, 4000, 40000, 4300, 7000
#             and 9000**
#         - ### Key Instruction: If any numbers from previous criteria like '4000012738', '4000095050' are found in
#         email
#             subject only are not just followed by 'Invoice' word, do not consider it as a Invoice Number.
#         - ### Key Instruction: If any number that starts with 160 like '1605083815', '1603912235' is found in email
#              subject or body and is followed by 'Invoice' word, do not consider it as a Invoice Number.
#     3. ### Key Instruction: Multiple Matches:
#        - If multiple valid **invoice numbers** are present, return all of them.
#     4. ### Key Instruction: No Match:
#         - If no valid invoice number is found, return an empty result.
#
# - **E-Invoice Number:**
#   - ### Key Instruction: **Pattern:** Always Start with E and never I.
#   - **Rule:** 10 Digits
#   - **Format:** Alpha Numeric
#   - **Comment:** E-invoices are uncertain in some business units, but the format starts with "E" and consists of
#   10 digits.
#
# Output:
# - ### Key Instruction: Return the response in the following JSON format. Do not add anything else
#     like comments/summary/messages.
#
# {{"response": {{"invoice_number": "<Extracted Invoice Number>", "e-invoice_number": "<Extracted E-Invoice Number>"}}}}
# """
#
# PROMPT_TEMPLATE_EXTRACT_PO_NUMBER = """
# You are an AI information extractor.
# - Your task is to extract the Purchase Order(PO) Number from the given email subject and body.
# - The email consists of details or concerns about a purchase order and may include various fields such as PO number,
#     invoice number, E-invoice number, vendor number, and the date etc.
# - You are tasked to extract the Purchase Order(PO) Number only.
# - ### Key Instruction: Analyze the content carefully before extracting anything.
#
# Below is the the subject and body of the email:
#
# {subject}: Subject of email
# {body}: Content/body of email
#
# - **Rules to Identify Purchase Order(PO) Number:**
#   - ### Key Instruction: It will Only Start with 45, 47 or 65. Do not pick any number which doesn't follow this
#     criteria. We don't want that number. Like 4000654565, 4300001568 is not a PO number.
#   - **Rule:** Strictly a 10 Digit number.
#   - **Format:** Strictly Numeric and can have special characters like "/" anywhere between 10 digits.
#   - ### Key Instruction: Multiple Matches:
#      - If multiple valid **PO numbers** are present, return all of them.
#   - ### Key Instruction: No Match:
#     - If no valid PO number is found, return an empty result.
#
# Output:
# - Return the response in the following JSON format. Do not add anything else including comments/messages.
#
# {{"response": {{"po_number": "<Extracted PO Number>"}}}}
# """


# PROMPT_TEMPLATE_EXTRACT_INVOICE_NUMBER = """
# You are an AI information extractor.
# - Your task is to extract the Invoice Number and E-Invoice Number from the given email subject and body.
# - The email consists of details or concerns about a purchase order and may include various fields such as PO number,
#   invoice number, E-invoice number, vendor number, and date, etc.
# - Your job is to extract **only the Invoice Number and E-Invoice Number**.
# - ### Key Instruction: Analyze the content carefully before extracting anything. Do not assume or hallucinate values.
#
# Below is the subject and body of the email:
#
# {subject}: Subject of email
# {body}: Content/body of email
#
# ---
#
# ### Rules to Identify **Invoice Number**:
# 1. **Format and Pattern**:
#    - Invoice Number may be **purely numeric** (e.g., '288617') or **alphanumeric** (e.g., 'TP0000330').
#    - It may start with **"I"**, **"IE"**, or **"INV"**, followed by numbers (e.g., "I3453423", "IE87271",
#    "INV1121342112").
#    - It may also include **"INV"** or **"IN"** in the middle of the number.
#    - Allowed special characters are **slash ("/")**, **hyphen ("-")**, **underscore ("_")**, or **dot (".")**.
#    - No spaces or other special symbols are allowed.
#
# 2. **Exclusions**:
#    - Do not extract any number starting with:
#      - **40, 45, 41, 43, 65, 800, 2000, 3000, 400, 4000, 40000, 4300, 7000, 9000**.
#    - If such numbers (e.g., '4000012738', '4000095050') appear in the **email subject** and are **not followed by
#      the word "Invoice"**, exclude them.
#    - Do not extract numbers starting with **160** (e.g., '1605083815', '1603912235') even if followed by the word
#      "Invoice" in the **subject or body**.
#
# 3. **Multiple Matches**:
#    - If multiple valid Invoice Numbers are present, extract all of them.
#
# 4. **No Match Scenario**:
#    - If no valid Invoice Number is found, return an empty result.
#
# ---
#
# ### Rules to Identify **E-Invoice Number**:
# 1. **Format**:
#    - E-Invoice Number always starts with the letter **"E"**.
#    - It must be **10 characters long** and strictly **alphanumeric**.
#    - E-Invoice Numbers never start with **"I"**.
#
# 2. **Handling Uncertainty**:
#    - If no valid E-Invoice Number is found in the email, return an empty result.
#
# ---
#
# ### Output Format:
# - Return the response in the following JSON format. Do not add anything else such as comments, summaries, or
# extra text.
#
# {{"response": {{"invoice_number": "<Extracted Invoice Number(s)>",
# "e-invoice_number": "<Extracted E-Invoice Number(s)>"}}}}
#
# """


PROMPT_TEMPLATE_EXTRACT_INVOICE_NUMBER = """
You are an AI information extractor.
- Your task is to extract the Invoice Number and E-Invoice Number from the given email subject and body.
- The email consists of details or concerns about a purchase order and may include various fields such as PO number, 
  invoice number, E-invoice number, vendor number, and date, etc.
- Your job is to extract **only the Invoice Number and E-Invoice Number**.
- ### Key Instruction: Always analyze the email **subject first** to locate the numbers. If not found, analyze the 
    email body. 
- ### Key Instruction: If you are not sure then do not assume or hallucinate values. Return an empty result.

Below is the subject and body of the email:

{subject}: Subject of email  
{body}: Content/body of email  

---

### Rules to Identify **Invoice Number**:
1. **Search Priority**:
   - First, search for the Invoice Number in the **email subject**. If not found, 
   proceed to search in the **email body**.

2. **Format and Pattern**:
   - Invoice Number may be **purely numeric** (e.g., '288617') or **alphanumeric** (e.g., 'TP0000330').
   - It may start with **"I"**, **"IE"**, or **"INV"**, followed by numbers (e.g., "I3453423", "IE87271", 
     "INV1121342112").
   - It may also include **"INV"** or **"IN"** somewhere in the middle of the number.
   - Allowed special characters are **slash ("/")**, **hyphen ("-")**, **underscore ("_")**, or **dot (".")**.
   - No spaces or other special symbols are allowed.

3. **Exclusions**:
   - Do not extract any number starting with:
     - **40, 45, 41, 43, 65, 800, 2000, 300, 3000, 400, 4000, 40000, 4300, 7000, 9000**. 
   - Do not extract numbers starting with **160** (e.g., '1605083815', '1603912235') even if followed by the word 
     "Invoice" in the **subject or body**.
     

4. **Multiple Matches**:
   - If multiple valid Invoice Numbers are present, extract all of them.

5. **No Match Scenario**:
   - If no valid Invoice Number is found, return an empty result.
   - ### Key Instruction: If you are not sure then do not assume or hallucinate values. Return an empty result.

---

### Rules to Identify **E-Invoice Number**:
1. **Search Priority**:
   - First, search for the E-Invoice Number in the **email subject**. If not found, proceed to search in the **email body**.

2. **Format**:
   - E-Invoice Number always starts with the letter **"E"**.
   - It must be **10 characters long** and strictly **alphanumeric**.
   - E-Invoice Numbers never start with **"I"**.

3. **Handling Uncertainty**:
   - If no valid E-Invoice Number is found in the email, return an empty result.

---

### Rules to Identify **Other Number**:
1. **Search Priority**:
   - First, search for the E-Invoice Number in the **email subject**.
     If not found, proceed to search in the **email body**.
2. If any number is found and you are not sure about that number then consider it as "Other number" and return it
    in the result.
3. Make sure that the other number strictly follows the **Exclusions** rules mentioned above.
---

### Output Format:
- Return the response in the following JSON format. Do not add anything else such as comments, summaries, or extra text.

{{"response": {{"invoice_number": "<Extracted Invoice Number(s)>", 
"e-invoice_number": "<Extracted E-Invoice Number(s)>", "other_number": <Extracted Other Number(s)>}}}}

"""
################################################################################################
# PROMPT_TEMPLATE_EXTRACT_PO_NUMBERt = """
# You are an AI information extractor.
# - Your task is to extract the Purchase Order(PO) Number from the given email subject and body.
# - The email consists of details or concerns about a purchase order and may include various fields such as PO number,
#     invoice number, E-invoice number, vendor number, and the date etc.
# - You are tasked to extract the Purchase Order(PO) Number only.
# - ### Key Instruction: Analyze the content carefully before extracting anything.
#
# Below is the the subject and body of the email:
#
# {subject}: Subject of email
# {body}: Content/body of email
#
# - Rules to Identify Purchase Order(PO) Number:
#     - Rule: Strictly a 10 Digit number.
#     - First two digits must be 45, 47 or 65. Do not extract any number which doesn't start with these three.
#     - Format: Strictly Numeric and can have special characters like "/" anywhere between 10 digits. Do not extract
#         Alphanumeric numbers.
#     - Multiple Matches:
#       - If multiple valid PO numbers are present, return all of them.
#     - No Match:
#       - If no valid PO number is found, return an empty result.
#
# Output:
# - Return the response in the following JSON format. Do not add anything else including comments/messages.
#
# {{"response": {{"po_number": "<Extracted PO Number>"}}}}
# """


# PROMPT_TEMPLATE_EXTRACT_PO_NUMBERt = """
# You are an AI information extractor.
# - Your task is to extract the Purchase Order (PO) Number from the given email subject and body.
# - The email consists of details or concerns about a purchase order and may include various fields such as PO number,
#   invoice number, E-invoice number, vendor number, and the date, etc.
# - Your task is to extract **only the Purchase Order (PO) Number**.
# - ### Key Instruction: Analyze the content carefully before extracting anything.
#
# Below is the subject and body of the email:
#
# {subject}: Subject of email
# {body}: Content/body of email
#
# ### Rules to Identify Purchase Order (PO) Number:
# 1. **Length and Format:**
#    - The PO number must be strictly **10 digits long**.
#    - It can include special characters like `/` between the digits (e.g., `45/12345678`).
#    - **Do not extract alphanumeric numbers** (e.g., `45ABC12345` is invalid).
#
# 2. **Starting Pattern:**
#    - The first two digits of any PO number must always **start with 45, 47, or 65**.
#    - Do not extract numbers that start with anything else.
#
# 3. **Exclusion Rules:**
#    - **Exclude any number** that:
#      - Starts with patterns such as `40`, `43`, `430`, or `400`.
#      - Is shorter or longer than 10 digits (excluding special characters).
#    - **Example of invalid numbers to exclude:**
#      - `4300123456`, `4000987654`, `4300/123456`, `4000/000001`.
#
# 4. **Handling Multiple Matches:**
#    - If multiple valid PO numbers are present, return all of them as a list.
#    - Ensure all extracted numbers follow the above rules strictly.
#
# 5. **No Match Scenario:**
#    - If no valid PO number is found, return an empty result.
#
# ### Output Format:
# - Return the response in the following JSON format. Do not add anything else, including comments or messages.
#
# {{"response": {{"po_number": "<Extracted PO Number(s)>"}}}}
#
# """

PROMPT_TEMPLATE_EXTRACT_PO_NUMBER = """
You are an AI information extractor. 
- Your task is to extract the Purchase Order (PO) Number from the given email subject and body. 
- The email consists of details or concerns about a purchase order and may include various fields such as PO number, 
  invoice number, E-invoice number, vendor number, and the date, etc. 
- Your task is to extract **only the Purchase Order (PO) Number**.
- ### Key Instruction: Always analyze the email **subject first** to locate the PO number. If not found, analyze the email body.

Below is the subject and body of the email:

{subject}: Subject of email  
{body}: Content/body of email

### Rules to Identify Purchase Order (PO) Number:
1. **Search Priority:**
   - First, search for the PO number in the **email subject**. If not found, proceed to search in the **email body**.

2. **Length and Format:**
   - The PO number must be strictly **10 digits long**.
   - The PO number must be strictly numeric.
   - It can include special characters like `/` between the digits (e.g., `45/12345678`).
   - **Do not extract alphanumeric numbers** (e.g., `45ABC12345` is invalid).

3. **Starting Pattern:**
   - The first two digits of any PO number must always **start with 45, 47, or 65**.
   - Do not extract numbers that start with anything else.

4. **Exclusion Rules:**
   - **Exclude any number** that:
     - Starts with patterns such as `40`, `43`, `430`, or `400`.
     - Is shorter or longer than 10 digits (excluding special characters).
   - **Example of invalid numbers to exclude:**
     - `4300123456`, `4000987654`, `4300/123456`, `4000/000001`.

5. **Handling Multiple Matches:**
   - If multiple valid PO numbers are present, return all of them as a list.
   - Ensure all extracted numbers follow the above rules strictly.

6. **No Match Scenario:**
   - If no valid PO number is found in both the subject and body, return an empty result.

---

### Output Format:
- Return the response in the following JSON format. Do not add anything else, including comments or messages.

{{"response": {{"po_number": "<Extracted PO Number(s)>"}}}}


"""

##################################################################################################


#   - ### Key Instruction: If any number that follows above criteria is found and is followed/Preceded by a
#     Firm/Company/Individual name in the email subject then it is a Vendor Number for sure.

# PROMPT_TEMPLATE_EXTRACT_VENDOR_NUMBER = """
# You are an AI information extractor.
# - Your task is to extract the Vendor Name and Vendor Number from the given email subject and body.
# - The email consists of details or concerns about a purchase order and may include various fields such as PO number,
#     invoice number, E-invoice number, vendor number, and the date etc.
# - You are tasked to extract the Vendor Name and Vendor Number only.
# - ### Key Instruction: Analyze the content carefully before extracting anything.
#
# Below is the the subject and body of the email:
#
# {subject}: Subject of email
# {body}: Content/body of email
#
# - **Vendor Number:**
#   - ### Key Instruction: Strictly a 10 Digits number.
#   - ### Key Instruction: Always Starts with either 43, 430, 4300, 43000, 40, 400, 4000, 40000, 200, 300, 700, 7000 or
#     90000 only. Sometimes It can also start with alphabet "P".
#   - ### Key Instruction: **Format:** Numeric only. Do not consider any alphanumeric number as Vendor number.
#   - ### Key Instruction: Multiple Matches:
#      - If multiple valid **Vendor Number** are present, return all of them.
#   - ### Key Instruction: No Match:
#     - If no valid Vendor Number is found, return an empty result.
#
# - **Vendor Name:**
#   - ### Key Instruction: First Check for Vendor Name in the email subject only. If not found then search in email body.
#   - **Pattern:** It should be a Firm/Company/Individual name.
#   - ### Key Instruction: Exclude all the names if it has "Coca-Cola" mentioned in it. It is strictly not
#     a vendor name.
#   - ### Key Instruction: **Rule:** Vendor Name is always followed/Preceded by a 10 Digits Vendor Number which
#     Always Starts with either 43, 430, 4300, 43000, 40, 400, 4000, 40000, 200, 300, 700, 7000 or
#     90000 only.
#   - **Format:** Only Text. Can be Abbreviations also.
#   - ### Key Instruction: Multiple Matches:
#     - If multiple valid **Vendor Name** are present, return all of them.
#   - ### Key Instruction: No Match:
#     - If no valid Vendor Name is found, return an empty result.
#
# Output:
# - ### Key Instruction: Return the response in the following JSON format. Do not add anything else
#     like comments/summary/messages.
#
# {{"response": {{"vendor_number": "<Extracted Vendor Number>", "vendor_name": "<Extracted Vendor Name>"}}}}
# """


# PROMPT_TEMPLATE_EXTRACT_VENDOR_NUMBER = """
# You are an AI information extractor.
# - Your task is to extract the Vendor Name and Vendor Number from the given email subject and body.
# - The email consists of details or concerns about a purchase order and may include various fields such as PO number,
#   invoice number, E-invoice number, vendor number, and date, etc.
# - Your job is to extract **only the Vendor Name and Vendor Number**.
# - ### Key Instruction: Analyze the content carefully before extracting anything.
#
# Below is the subject and body of the email:
#
# {subject}: Subject of email
# {body}: Content/body of email
#
# ### Rules to Identify **Vendor Number**:
# 1. **Length and Format**:
#    - Vendor Number must be strictly **10 digits long**.
#    - It can only be **numeric** and cannot include alphanumeric characters.
#
# 2. **Starting Pattern**:
#    - Vendor Number must always start with one of the following:
#      - **43, 430, 4300, 43000, 40, 400, 4000, 40000, 200, 300, 700, 7000, 90000**
#      - **OR start with the alphabet "P".**
#    - Do not extract numbers starting with any other pattern.
#
# 3. **Multiple Matches**:
#    - If multiple valid Vendor Numbers are present, return all of them in the response.
#
# 4. **No Match Scenario**:
#    - If no valid Vendor Number is found, return an empty result.
#
# ---
#
# ### Rules to Identify **Vendor Name**:
# 1. **Source**:
#    - First, check for the Vendor Name in the **email subject**. If not found, search the **email body**.
#
# 2. **Format**:
#    - Vendor Name must be a proper **Firm/Company/Individual name**.
#    - It can also be abbreviations but cannot include invalid patterns like numerical sequences.
#
# 3. **Exclusion Rules**:
#    - **Exclude all names** that contain the word **"Coca-Cola"**, as this is not considered a vendor name.
#
# 4. **Relation to Vendor Number**:
#    - A Vendor Name is usually **preceded or followed** by a valid **10-digit Vendor Number** starting with:
#      - **43, 430, 4300, 43000, 40, 400, 4000, 40000, 200, 300, 700, 7000, 90000, or "P".**
#
# 5. **Multiple Matches**:
#    - If multiple valid Vendor Names are found, return all of them in the response.
#
# 6. **No Match Scenario**:
#    - If no valid Vendor Name is found, return an empty result.
#
# ---
#
# ### Output Format:
# - Return the response in the following JSON format. Do not add anything else like comments, summaries, or extra text.
#
# {{"response": {{"vendor_number": "<Extracted Vendor Number(s)>", "vendor_name": "<Extracted Vendor Name(s)>"}}}}
#
# """


PROMPT_TEMPLATE_EXTRACT_VENDOR_NUMBER = """

You are an AI information extractor. 
- Your task is to extract the Vendor Name and Vendor Number from the given email subject and body. 
- The email consists of details or concerns about a purchase order and may include various fields such as PO number, 
  invoice number, E-invoice number, vendor number, and date, etc. 
- Your job is to extract **only the Vendor Name and Vendor Number**.
- ### Key Instruction: Analyze the content carefully and prioritize checking the **email subject first** for Vendor 
Number before analyzing the email body.

Below is the subject and body of the email:

{subject}: Subject of email  
{body}: Content/body of email


### Rules to Identify **Vendor Number**:
1. **Search Priority**:
   - First, search for the Vendor Number in the **email subject**. If not found, proceed to search in the **email body**.

2. **Length and Format**:
   - Vendor Number must be strictly **10 digits long**.
   - It can only be **numeric** and cannot include alphanumeric characters. 

3. **Starting Pattern**:
   - Vendor Number must always start with one of the following:
     - **43, 430, 4300, 43000, 40, 400, 4000, 40000, 200, 300, 700, 7000, 90000**
     - **OR start with the alphabet "P".**
     
3. **Exclusion Rules**:
   - Do not extract numbers starting with '45', '47', '65' & '800'.
   - Do not extract any other pattern other than rules mentioned in starting pattern.

4. **Multiple Matches**:
   - If multiple valid Vendor Numbers are present, return all of them in the response.

5. **No Match Scenario**:
   - If no valid Vendor Number is found in both the subject and body, return an empty result.

---

### Rules to Identify **Vendor Name**:
1. **Search Priority**:
   - First, check for the Vendor Name in the **email subject**. If not found, search the **email body**.

2. **Format**:
   - Vendor Name must be a proper **Firm/Company/Individual name**.
   - It can also be abbreviations but cannot include invalid patterns like numerical sequences.

3. **Exclusion Rules**:
   - **Exclude all names** that contain the word **"Coca-Cola"**, as this is not considered as a vendor name.

4. **Relation to Vendor Number**:
   - A Vendor Name is usually **preceded or followed** by a valid **10-digit Vendor Number** starting with:
     - **43, 430, 4300, 43000, 40, 400, 4000, 40000, 200, 300, 700, 7000, 90000, or "P".**

5. **Multiple Matches**:
   - If multiple valid Vendor Names are found, return all of them in the response.

6. **No Match Scenario**:
   - If no valid Vendor Name is found in both the subject and body, return an empty result.

---

### Output Format:
- Return the response in the following JSON format. Do not add anything else like comments, summaries, or extra text.

{{"response": {{"vendor_number": "<Extracted Vendor Number(s)>", "vendor_name": "<Extracted Vendor Name(s)>"}}}}


Below is the subject and body of the email as Short description and Description::
 
"""


# PROMPT_TEMPLATE_FINAL_RESPONSE = """
# - You are an intelligent assistant tasked with generating a email response to a Purchase Order (PO) or
# Invoice-related email.
# - Based on the provided email body, create a polite, professional, and appropriate reply that addresses the key points,
# queries, or requests related to the PO or invoice.
#
# Below is the the subject and body of the email:
#
# {subject}: Subject of email
# {body}: Content/body of email
#
# Below are the details to use if needed in response.
# {
#     {Internal Invoice Number}: Unique identification number assigned to the internal invoice,
#     {External Invoice Number}: Unique identifier for the external invoice,
#     {Vendor Number}: Unique identifier for the supplier or vendor,
#     {PO Number}: Unique purchase order number related to the transaction,
#     {Invoice Issuing Date}: The date on which the invoice is issued,
#     {Company Code}: Unique code assigned to the company for accounting purposes,
#     {Invoice Amount}: The total amount to be paid for the invoice, including taxes or adjustments,
#     {Invoice Currency}: The currency in which the invoice is issued (e.g., Euro, USD),
#     {Posting Date}: The date when the transaction or invoice is recorded in the system,
#     {Payment Date}: The date by which payment for the invoice is due,
#     {Payment Document Number}: Unique identifier for the payment document,
#     {Workflow Status}: The current status of the invoice in the workflow (e.g., Open, Approved, Paid),
#     {Workflow Description}: Additional details or explanation of the workflow status,
#     {Invoice Due Date}: The date by which the payment must be completed,
#     {Invoice Status}: The current state of the invoice, such as Pending, Paid, or Overdue,
#     {Block Reason}: Reason for any block or restriction on the invoice or payment
# }
#
# Instructions:
#
# - Carefully review the email body and identify any PO or invoice-related details, such as requests for updates,
#     confirmations, discrepancies, payment status, or delivery issues.
# - Craft a reply that:
#     - Acknowledges the sender’s request or concern.
#     - Provides clear and concise information about the status of the PO or invoice, or addresses any discrepancies.
#     - Confirms any required actions, such as payment processing, shipping details, or further follow-up.
#     - Includes any relevant dates, amounts, or other PO/invoice details as needed.
#     - Ensure the tone is professional, with formal or semi-formal language based on the context of the email.
#     - Add original email body in the response
#
# - Return the response in the following JSON format only:
#
# {{"response": {{"response_body": "<Your generated response here including Original email body>",
# "tone": "<formal/semi-formal>"}}}}
# """

PROMPT_TEMPLATE_FINAL_RESPONSE = """
- You are an intelligent assistant tasked with generating a email response to a Purchase Order (PO) or 
Invoice-related email. 
- Based on the provided email body, create a polite, professional, and appropriate reply that addresses the key points, 
queries, or requests related to the PO or invoice.

Below is the the subject and body of the email including some other details. Use these details if needed 
while generating the response. 
{
    {subject}: Subject of email
    {body}: Content/body of email
    {Internal Invoice Number}: Unique identification number assigned to the internal invoice,  
    {External Invoice Number}: Unique identifier for the external invoice,  
    {Vendor Number}: Unique identifier for the supplier or vendor,  
    {PO Number}: Unique purchase order number related to the transaction,  
    {Invoice Issuing Date}: The date on which the invoice is issued,  
    {Company Code}: Unique code assigned to the company for accounting purposes,  
    {Invoice Amount}: The total amount to be paid for the invoice, including taxes or adjustments,  
    {Invoice Currency}: The currency in which the invoice is issued (e.g., Euro, USD),  
    {Posting Date}: The date when the transaction or invoice is recorded in the system,  
    {Payment Date}: The date by which payment for the invoice is due,  
    {Payment Document Number}: Unique identifier for the payment document,  
    {Workflow Status}: The current status of the invoice in the workflow (e.g., Open, Approved, Paid),  
    {Workflow Description}: Additional details or explanation of the workflow status,  
    {Invoice Due Date}: The date by which the payment must be completed,  
    {Invoice Status}: The current state of the invoice, such as Pending, Paid, or Overdue,  
    {Block Reason}: Reason for any block or restriction on the invoice or payment
}

Instructions:

- Carefully review the email body and identify any PO or invoice-related details, such as requests for updates, 
    confirmations, discrepancies, payment status, or delivery issues.
- Craft a reply that:
    - Acknowledges the sender’s request or concern.
    - Provides clear and concise information about the status of the PO or invoice, or addresses any discrepancies.
    - Confirms any required actions, such as payment processing, shipping details, or further follow-up.
    - Includes any relevant dates, amounts, or other PO/invoice details as needed.
    - Ensure the tone is professional, with formal or semi-formal language based on the context of the email.
    - Add original email body in the response

- Return the response in the following JSON format only:

{{"response": {{"response_body": "<Your generated response here including Original email body>", 
"tone": "<formal/semi-formal>"}}}}
"""


##################################################################################################

# PROMPT_TEMPLATE_SIGNATURE = """
# 1. Task:
# - Extract all email signatures from the provided email body text. The email body may include an email chain
# (multiple email replies).
# - Email signatures are located at the end of each individual message and may include elements such as the sender's
# name, job title, company name, contact details, and disclaimers.
# - Provide the output in a structured and consistent format.
#
# Below is the the subject and body of the email:
#
# {subject}: Subject of email
# {body}: Content/body of email
#
# 2. Instructions:
#
# - Identify Email Signatures:
#   - Look for sections at the end of individual messages in the email chain that resemble signature blocks.
# - Signature blocks often contain:
#   - Sender's name.
#   - Job title or role.
#   - Company name or logo reference.
#   - Contact details (e.g., email address, phone number, website).
#   - Additional details (e.g., physical address, disclaimers, social media links).
#   - Exclude Non-Signature Text:
#   - Do not include the main email content, quoted text, or forwarded email headers.
#   - Ignore greetings (e.g., "Hi John") or sign-offs without a signature (e.g., "Best regards" alone).
#
# 3. Output Format:
#   - List each extracted signature separately.
#   - Use the format Signature X where X is the sequence number of the email in the chain.
#   - Maintain the order of signatures as they appear in the email body.
#   - If no signature is present in an email, skip it.
#
# 4. Few-Shot Examples:
#
# Example 1: Single Email
# Input:
# Hi Team,
#
# Please review the attached file and let me know your feedback.
#
#
# Jane Doe
# Senior Manager
# ABC Corporation
# jane.doe@abc.com
# +1-234-567-8900
#
# ---
#
# Hi Jane,
#
# I’ve attached the updated document. Let me know if it works.
#
#
# John Smith
# Project Lead
# XYZ Ltd.
# john.smith@xyz.com
# +1-987-654-3210
# www.xyz.com
# -----
#
# Output:
#
# {{"response": {{"Signature 1": "Jane Doe
# Senior Manager
# ABC Corporation
# jane.doe@abc.com
# +1-234-567-8900",
#
# "Signature 2": <John Smith
# Project Lead
# XYZ Ltd.
# john.smith@xyz.com
# +1-987-654-3210
# www.xyz.com >,
#
# - ### Key Instruction: Multiple Matches:
#     - If multiple valid **Signatures** are present, return all of them.
# - ### Key Instruction: No Match:
#     - If no valid **Signature** is found, return an empty result.
# """


PROMPT_TEMPLATE_SIGNATURE = """
Task: 
Extract email signatures from the provided email body. 
Email signatures are located at the end of individual messages in the email chain and may include:
- Sender's name.
- Job title or role.
- Company name.
- Contact details (email, phone, website).
- Additional details like physical address, disclaimers, or social media links.

Below is the body of the email:

{body}: Content/body of email

Instructions:
2. Identify and extract signature blocks from the end of each message in the chain.
3. Exclude non-signature text such as:
   - Main email content, greetings (e.g., "Hi John"), and generic sign-offs (e.g., "Best regards").
   - Quoted text or forwarded email headers.
4. List extracted signatures in sequence as they appear in the email body.
5. If no signature is found in a message, skip it.

Output Format:
- Return the extracted signatures as JSON with each signature labeled as "Signature X", where X is the sequence number.
- If no signature is found, return an empty result.

Example Output:
{{"response": {{"Signature X": <Extracted Signature>}}}}
"""


###########################################################################################################

# COMBINED_PROMPT = """
#
# You are an AI information extractor.
# - Your task is to extract the following information from the given email subject and body:
#   1. **Vendor Name** and **Vendor Number**
#   2. **Purchase Order (PO) Number**
#   3. **Invoice Number** and **E-Invoice Number**
#
# - The email consists of details or concerns about a purchase order and may include fields such as PO number, invoice number, E-invoice number, vendor number, vendor name, and date, etc.
# - Extract only the specified details while ensuring that a number is **classified exclusively** (e.g., a number cannot be both a Vendor Number and an Invoice Number).
#
# Below is the body of the email:
#
# {subject}: Subject of the email
# {body}: Content/body of email
#
# ---
#
# ## General Rules:
# 1. **Order of Extraction**:
#    - First extract **Vendor Name** and **Vendor Number**.
#    - Then extract **PO Number**.
#    - Finally, extract **Invoice Number** and **E-Invoice Number**.
#
# 2. **Exclusion of Duplicate Numbers**:
#    - Once a number is classified (e.g., as a Vendor Number), it must not be reused or extracted under another category (e.g., as a PO or Invoice Number).
#
# 3. **Priority**:
#    - Always analyze the **email subject first**. If the required information is not found, analyze the **email body**.
#
# 4. **Uncertainty**:
#    - If unsure about any number or name, return an empty result. Do not assume or hallucinate values.
#
# ---
#
# ## Extraction Rules:
#
# ### **Vendor Name and Vendor Number**
# 1. **Vendor Number Rules**:
#    - **Format and Starting Pattern**:
#      - Must be strictly **10 digits long**.
#      - Can only be numeric.
#      - Must start with: `43, 430, 4300, 43000, 40, 400, 4000, 40000, 200, 300, 700, 7000, 90000, or "P"`.
#      - Exclude numbers starting with `45, 47, 65, 800`.
#
#    - **Multiple Matches**: Extract all valid Vendor Numbers.
#
# 2. **Vendor Name Rules**:
#    - Must be a proper **Firm/Company/Individual name**.
#    - Usually **preceded or followed** by a valid Vendor Number.
#    - Exclude names containing invalid patterns (e.g., "Coca-Cola").
#
# ---
#
# ### **Purchase Order (PO) Number**
# 1. **PO Number Rules**:
#    - **Format and Starting Pattern**:
#      - Must be strictly **10 digits long** (numeric only).
#      - May include special characters like `/` (e.g., `45/12345678`).
#      - Must start with `45, 47, or 65`.
#      - Exclude numbers starting with `40, 43, 430, 400`.
#
#    - **Search Priority**:
#      - Search in the **subject** first; if not found, search in the **body**.
#
#    - **Multiple Matches**: Extract all valid PO Numbers.
#
# ---
#
# ### **Invoice Number and E-Invoice Number**
# 1. **Invoice Number Rules**:
#    - **Format and Pattern**:
#      - May be purely numeric (e.g., `288617`) or alphanumeric (e.g., `TP0000330`).
#      - May start with: `I, IE, or INV` followed by numbers (e.g., `I3453423`, `INV1121342112`).
#      - Allowed special characters: `/`, `-`, `_`, or `.`.
#
#    - **Exclusions**:
#      - Exclude numbers starting with: `40, 45, 41, 43, 65, 800, 2000, 300, 3000, 400, 4000, 40000, 4300, 7000, 9000`.
#      - Exclude numbers starting with `160`, even if preceded or followed by the word "Invoice".
#
# 2. **E-Invoice Number Rules**:
#    - **Format**:
#      - Must start with the letter **"E"**.
#      - Must be exactly **10 characters long** and strictly alphanumeric.
#      - Cannot start with **"I"**.
#
# 3. **Other Numbers**:
#    - Numbers not classified as Vendor, PO, or Invoice Numbers can be considered as **Other Numbers**, following all exclusion rules.
#
# ---
#
# ## Output Format:
# Return the extracted information in the following JSON format. Do not include any extra comments or text.
#
# {{
#   "response": {{
#     "vendor_name": "<Extracted Vendor Name(s)>",
#     "vendor_number": "<Extracted Vendor Number(s)>",
#     "po_number": "<Extracted PO Number(s)>",
#     "invoice_number": "<Extracted Invoice Number(s)>",
#     "e-invoice_number": "<Extracted E-Invoice Number(s)>",
#     "other_number": "<Extracted Other Number(s)>"
#   }}
# }}
#
# """


# COMBINED_PROMPT = """
#
# You are a meticulous data reader and extractor with a keen eye for detail and a mastery of the different languages.
# Your goal is to thoroughly review the provided email draft text first.
# - Your task is to extract the following information from the given email subject and body:
#   1. **Vendor Name** and **Vendor Number**
#   2. **Purchase Order (PO) Number**
#   3. **Invoice Number** and **E-Invoice Number**
#
# - The email consists of details or concerns about a purchase order and may include fields such as PO number, invoice
# number, E-invoice number, vendor number, vendor name, and date, etc.
# - Extract only the specified details while ensuring that a number is **classified exclusively**.
#
# Below is the subject and body of the email:
#
# {subject}: Subject of the email
# {body}: Content/body of email
#
# ---
#
# ## General Rules:
# 1. **Order of Extraction**:
#    - First extract **Vendor Name** and **Vendor Number**.
#    - Then extract **PO Number**.
#    - Finally, extract **Invoice Number** and **E-Invoice Number**.
#
# 2. **Exclusion of Duplicate Numbers**:
#    - Once a number is classified (e.g., as a Vendor Number), it must not be reused or extracted under another category
#    (e.g., as a PO or Invoice Number).
#
# 3. **Priority**:
#    - Always analyze the **email subject first**. If the required information is not found, analyze the **email body**.
#
# 4. **Uncertainty**:
#    - If unsure about any number or name, return an empty result. Do not assume or hallucinate values.
#
# ---
#
# ## Extraction Rules:
#
# ### **Vendor Name and Vendor Number**
# 1. **Vendor Number Rules**:
#    - **Format and Starting Pattern**:
#      - Must be strictly **10 digits long**.
#      - Can only be numeric.
#      - Must start with: `43, 430, 4300, 43000, 40, 400, 4000, 40000, 200, 300, 700, 7000, 90000, or "P"`.
#      - Exclude numbers starting with `45, 47, 65, 800, 900`.
#
#    - **Multiple Matches**: Extract all valid Vendor Numbers.
#
# 2. **Vendor Name Rules**:
#    - Must be a proper **Firm/Company/Individual name**.
#    - Usually **preceded or followed** by a valid Vendor Number.
#    - Exclude names containing invalid patterns (e.g., "Coca-Cola").
#
# ---
#
# ### **Purchase Order (PO) Number**
# 1. **PO Number Rules**:
#    - **Format and Starting Pattern**:
#      - Must be strictly **10 digits long** (numeric only).
#      - May include special characters like `/` (e.g., `45/12345678`).
#      - Must start with `45, 47, or 65`.
#      - Exclude numbers starting with `40, 43, 430, 400`.
#
#    - **Search Priority**:
#      - Search in the **subject** first; if not found, search in the **body**.
#
#    - **Multiple Matches**: Extract all valid PO Numbers.
#
# ---
#
# ### **Invoice Number and E-Invoice Number**
# 1. **Invoice Number Rules**:
#    - **Format and Pattern**:
#      - May be purely numeric (e.g., `288617`) or alphanumeric (e.g., `TP0000330`).
#      - May start with: `I, IE, or INV` followed by numbers (e.g., `I3453423`, `INV1121342112`).
#      - Allowed special characters: `/`, `-`, `_`, or `.`.
#
#    - **Exclusions**:
#      - Never ever consider above extracted Vendor Number & PO number as Invoice Number.
#      - Exclude numbers starting with `160`, even if preceded or followed by the word "Invoice".
#
# 2. **E-Invoice Number Rules**:
#    - **Format**:
#      - Must start with the letter **"E"**.
#      - Must be exactly **10 characters long** and strictly alphanumeric.
#      - Cannot start with **"I"**.
#
# 3. **Other Numbers**:
#    - Numbers not classified as Vendor, PO, or Invoice Numbers can be considered as **Other Numbers**,
#    following all exclusion rules.
#
# ---
#
# ## Output Format:
# Return the extracted information in the following JSON format. Do not include any extra comments or text.
#
# {{
#   "response": {{
#     "vendor_name": "<Extracted Vendor Name(s)>",
#     "vendor_number": "<Extracted Vendor Number(s)>",
#     "po_number": "<Extracted PO Number(s)>",
#     "invoice_number": "<Extracted Invoice Number(s)>",
#     "e-invoice_number": "<Extracted E-Invoice Number(s)>",
#     "other_number": "<Extracted Other Number(s)>"
#   }}
# }}
#
# """

# - Must start
# with one of these prefixes: `43`, `430`, `4300`, `43000`, `40`, `400`, `4000`, `40000`, `200`, `300`,
# `700`, `7000`, `90000`, or `"P"`.

COMBINED_PROMPT = """
You are a meticulous data reader and extractor with a keen eye for detail and a mastery of different languages.
Your goal is to **thoroughly review the provided email draft** text and **extract only specific information** with
absolute precision, following the strict rules outlined below.

## Email Structure:
- The email contains details or concerns about a purchase order and may include fields such as **PO number**,
    **invoice number**, **E-invoice number**, **vendor number**, **vendor name**, and **date**.
- **Extract only the specified details**—vendor name, vendor number, PO number, invoice number, and E-invoice number.
- Any **other number** that doesn't fit these categories should be classified as "Other number."

---

## Task:
Extract the following details from the **email subject and body**:
1. **Vendor Name** and **Vendor Number**
2. **Purchase Order (PO) Number**
3. **Invoice Number** and **E-Invoice Number**

---

Below is the subject and body of the email:

{subject}: Subject of the email
{body}: Content/body of email

---

## General Rules:
1. **Order of Extraction**:
   - First, extract **Vendor Name** and **Vendor Number**.
   - Then, extract **PO Number**.
   - Finally, extract **Invoice Number** and **E-Invoice Number**.

2. **Exclusion of Duplicate Numbers**:
   - **Do NOT reuse or classify the same number** in multiple categories (e.g., as both Vendor Number and PO Number).
   - A number must be classified **ONLY ONCE** and **must belong to one specific category** (Vendor, PO, or Invoice).
   - If a number is already classified as **Vendor Number**, **PO Number**, or **Invoice Number**, **DO NOT reclassify
    it under any other category**.

3. **Uncertainty**:
   - If you are **unsure** about any number or name, **return an empty result**.
   - Do not make assumptions or classify numbers incorrectly.

4. **Priority**:
   - **Always prioritize the email subject** to search for the required information.
   - If not found in the **subject**, then search the **body**.

---

## Extraction Rules:

### 1. Vendor Name and Vendor Number:
- **Vendor Number**:
   - Must be exactly **10 digits long** and **only numeric**.
   - **First two digits of the any vendor number** will always be `40`, `43`, `20`, `30`, `70` and `90`. 
   - **Exclude Vendor Numbers** where First two digits are either of `45`, `47`, `65`, `80`, or `90`.
   - **Extract all valid Vendor Numbers** found.

- **Vendor Name**:
   - Must be a **proper firm/company/individual name**.
   - Vendor Name is typically **preceded or followed by a valid Vendor Number**.
   - **Do NOT consider names containing invalid patterns**, such as "Coca-Cola" or names with symbols like "-".

### 2. Purchase Order (PO) Number:
- **PO Number**:
   - Must be exactly **10 digits long** and **numeric only** (may include special characters like `/` such as `45/12345678`).
   - **Must start with `45`, `47`, or `65`**.
   - **Exclude numbers starting with `40`, `43`, `430`, `400`**.
   - **Multiple valid PO numbers may exist**. Extract **all valid PO numbers**.

### 3. Invoice Number:
- **Invoice Number**:
   - May be **numeric only** (e.g., `288617`) or **alphanumeric** (e.g., `TP0000330`).
   - Can start with `I`, `IE`, or `INV` followed by numbers (e.g., `I3453423`, `IE7645434`,`INV1121342112`).
   - May include special characters such as `/`, `-`, `_`, or `.` (e.g., `INV-1234_5678`).
   - **DO NOT classify a Vendor Number or PO Number** as an Invoice Number.
   - **Exclude Invoice Numbers starting with `160`**.

### 4. E-Invoice Number:
- **E-Invoice Number**:
   - Must start with **"E"** and be exactly **10 characters long**.
   - The E-Invoice Number must be strictly **alphanumeric** (e.g., `E123456789`).
   - **DO NOT classify any Invoice Number as an E-Invoice Number**.
   - **E-Invoice Number cannot start with "I"**.

### 5. Other Numbers:
- Any number that **does not follow any of the above rules** (for Vendor, PO, Invoice, or E-Invoice Number)
    should be classified as **Other Numbers**. For example:
  - Numbers not starting with any valid prefix for Vendor, PO, or Invoice should be classified as "Other Number."

---

## Output Format:
Return the extracted information in the following **exact JSON format**:
{{
  "response": {{
    "vendor_name": "<Extracted Vendor Name(s)>",
    "vendor_number": "<Extracted Vendor Number(s)>",
    "po_number": "<Extracted PO Number(s)>",
    "invoice_number": "<Extracted Invoice Number(s)>",
    "e-invoice_number": "<Extracted E-Invoice Number(s)>",
    "other_number": "<Extracted Other Number(s)>"
  }}
}}

"""
# - **If in doubt**, return "Other Finance and Non-Finance Query" as a fallback category,
# which represents queries that do not strictly fit into the other categories.
# ### 4. **Other Finance and Non-Finance Query**:
#    - **Definition**: This category is a general one that covers emails not directly related to PO, payment, or invoices,
#    but still related to financial or operational concerns.
#    - **Indicators**:
#      - Emails that are **general financial inquiries**, but don’t fit the specific criteria for other categories.
#      - Could involve **financial reports**, **accounting inquiries**, or operational queries unrelated to POs or payments.
#      - Non-financial concerns could include **administrative issues**, **general inquiries** unrelated to specific transactional concerns.
#
#    - **Example Phrases**:
#      - "Can you provide a breakdown of our account balance?"
#      - "We need assistance with updating our contact information."
#      - "Please send the latest financial reports."


PROMPT_EMAIL_CATEGORIZATION = """
You are a expert data analyzer tasked with categorizing an email into one of the following predefined categories:
- **Red Letter/Reminder**
- **PO Query**
- **Payment Query**
- **Invoice Query**
- **Data Maintenance**
- **Other Finance and Non-Finance Query**:

Your task is to **read the subject and body of the email carefully** and classify it into the most appropriate category 
based on the content. Follow these rules strictly to ensure accuracy and avoid hallucination or mis-classification.

## General Instructions:
1. **Initial Analysis**: 
   - Start by analyzing the **subject** of the email.
   - If the subject does not contain sufficient information, proceed to analyze the **body** of the email.
   
2. **Only Classify Based on Clear Content**:
   - **Do not make assumptions** or attempt to infer missing information. Only classify based on the **explicit** content in the email.
   - If the email does not clearly fit any of the categories below, return an empty result.

3. **Return Empty Result**:
   - If the email cannot be classified clearly into one of the provided categories, return an empty result, not any substitute category.

---

Below is the subject and body of the email:

{subject}: Subject of the email
{body}: Content/body of email

---

## Categorization Guidelines:

### 1. **Red Letter/Reminder**:
   - **Definition**: This category is for emails that serve as **urgent reminders** or **final notices**. 
   They are generally high-priority communications emphasizing **overdue actions** or **unresolved issues** that need immediate attention to avoid negative consequences. These emails often reflect a sense of urgency or escalation.
   - **Characteristics**:
     - The email conveys **urgency** and requests **immediate action** or **response**.
     - It is typically related to an action that is **overdue** or needs to be completed **within a short time frame**.
     - The message may reference **escalations** or **consequences** if action is not taken, such as penalties or further delays.
     - The tone is formal and emphasizes the importance of responding promptly.

### 2. **PO Query** (Purchase Order Query):
   - **Definition**: This category pertains to emails that involve **questions**, **clarifications**, or issues related to **Purchase Orders (POs)**. It may include inquiries about **missing POs**, **incorrect PO details**, or **requests for updates** or **status checks** on POs.
   - **Characteristics**:
     - The email explicitly references a **Purchase Order number** or a **related transaction**.
     - It might inquire about the **approval**, **processing**, or **delivery status** of the PO.
     - The email may highlight discrepancies or ask for clarification on any **PO details** (such as items, quantities, or pricing).
     - It might also ask for **confirmation** regarding PO-related tasks or steps in the process.

### 3. **Payment Query**:
   - **Definition**: This category applies to emails that are related to issues concerning **payments** or **financial transactions**. It includes inquiries about **payment status**, **delays in payments**, **payment discrepancies**, or **confirmation of received payments**.
   - **Characteristics**:
     - The email refers to **payments** or **payment statuses**.
     - It may inquire about the **status of a pending payment**, **delayed payments**, or **payment confirmation** for specific invoices, POs, or contracts.
     - The email might also express concern about discrepancies in **payment amounts** or **payment receipt confirmations**.
     - This category may include requests for clarification on the **timing** or **processing of payments**.

### 4. **Other Finance and Non-Finance Query**:
   - **Definition**: This category covers emails that do not directly fit into the categories of **PO Query**, **Payment Query**, **Invoice Query**, or **Data Maintenance**. It includes general **inquiries related to finance** or **other operational matters** not strictly tied to transactional issues.
   - **Characteristics**:
     - The email may cover **general finance-related queries** that do not involve payments, invoices, or purchase orders.
     - It could involve questions about **accounting policies**, **financial reporting**, or **general budget inquiries**.
     - It may include **administrative issues** such as document processing, **financial document management**, or even **budget allocation** that doesn't directly relate to transactions or data.
     - For **non-financial matters**, it might involve general operational issues not specific to any of the categories above, such as **report generation**, **system updates**, or **other generic inquiries**.

### 5. **Invoice Query**:
   - **Definition**: This category refers to emails that inquire about or raise concerns regarding **invoices**. It may involve questions about the **invoice details**, **payment status**, discrepancies between **invoice amounts**, or missing invoices.
   - **Characteristics**:
     - The email mentions **invoice numbers**, **invoice amounts**, or **invoice due dates**.
     - The content may ask for **clarifications** about the **accuracy** of an invoice, such as incorrect amounts, **missing invoice details**, or **disputed charges**.
     - The email may request to **reissue invoices**, **resend invoice copies**, or **explain discrepancies** in the invoicing.
     - It could also include inquiries related to **invoice payment status** or request confirmation on whether the invoice has been paid.

### 6. **Data Maintenance**:
   - **Definition**: This category is for emails concerning the **maintenance, update, or modification** of **records or data**. This can include tasks such as **updating contact information**, **updating vendor records**, or correcting **data discrepancies** in internal systems or databases.
   - **Characteristics**:
     - The email refers to tasks related to the **updating of records** such as contact details, addresses, or **vendor information**.
     - It may inquire about **modifying or correcting** data entries in the system or on invoices.
     - The email might request **system changes** or ask for corrections in previously entered **data records**.
     - It may involve requests for **account information updates** or other administrative tasks related to maintaining accurate records.

---

## Output Format:
Return the **final category** of the email using the following JSON format:
{{
  "response": {{"category": "<Category Response from LLM>"}}
}}
"""


COMBINED_PROMPT_SN = """

You are a meticulous data reader and extractor with a keen eye for detail and a mastery of different languages. Your goal is to **thoroughly review the provided email draft** text and **extract only specific information** with absolute precision, following the strict rules outlined below.  You will find the input email subject and body at the end.

## Email Structure:
- The email contains details or concerns about a purchase order and may include fields such as **PO number**, **invoice number**, **E-invoice number**, **vendor number**, **vendor name**, and **date**.
- **Extract only the specified details**—vendor name, vendor number, PO number, invoice number, and E-invoice number.
-  Any **other number** that doesn't fit these categories should be classified as "Other number."

---

## Task:
Extract the following details from the **email subject and body**:

1. **Vendor Name** and **Vendor Number**
2. **Purchase Order (PO) Number**
3. **Invoice Number** and **E-Invoice Number**

---

## General Rules:
1. **Order of Extraction**:  
   - First, extract **Vendor Name** and **Vendor Number**.  
   - Then, extract **PO Number**.  
   - Finally, extract **Invoice Number** and **E-Invoice Number**.

2. **Exclusion of Duplicate Numbers**:  
   - A number must be classified **ONLY ONCE** and **must belong to one specific category** (Vendor, PO, or Invoice).  
   - If a number is already classified as **Vendor Number**, **PO Number**, or **Invoice Number**, **DO NOT reclassify it under any other category**.

3. **Uncertainty**:  
   - If you are **unsure** about any number go through the below rules again. If still not sure , **return an empty result**.  
   - Do not make assumptions or classify numbers incorrectly.

4. **Priority**:  
   - **Always prioritize the email subject** to search for the required information.  
   - If not found in the **subject**, then search the **body**.

5. **Language**:
    - If the language of the input text is Serbian, Greek or Bulgarian, then convert the text into English first and 
    then apply below rules.

---

Below is the subject and body of the email:

{subject}: Subject of the email
{body}: Content/body of email

---

## Extraction Rules:

### 1. Vendor Name and Vendor Number:
- **Vendor Number**:  
   - Must be exactly **10 digits long** and **only numeric**.  
   - **First two digits of any vendor number** will always be `40`, `43`, `20`, `30`, `70`, and `90`.  
   - **Exclude Numbers** where first two digits are either of `45`, `47`, `65`,  and '80' . 
   - **Double check email subject** to find the Vendor number. It is mostly present in email subject only.
   - **Look for all such numbers in email subject and body.  Extract all valid Vendor Numbers** found.

- **Vendor Name**:  
   - Must be a **proper firm/company/individual name**.  
   - Vendor Name is typically **preceded or followed by a valid Vendor Number**.  
   - **Do NOT consider names containing invalid patterns**, such as "Coca-Cola" or names with symbols like "-".

### 2. Purchase Order (PO) Number:
- **PO Number**:  
   - Must be exactly **10 digits long** and **numeric only** (may include special characters like `/` such as `45/12345678`).  
   - **Must start with `45`, `47`, or `65`**.  
   - **Exclude numbers with **First two digits** are `40`, `43`, `20`, `30`, `70`, and `90`.  
   - **Look for all such numbers in email subject and body.  Extract all valid PO Numbers** found.

### 3. Invoice Number:
- **Invoice Number**:  
   - Important: First detect the language of the text. If it is in Serbian, Bulgaria or Greek, convert it into english first then apply below rules.
   - May be **numeric only** (e.g., `288617`) or **alphanumeric** (e.g., `TP0000330`).  
   - Can start with `I`, `IE`, or `INV` followed by numbers (e.g., `I3453423`, `IE7645434`, `INV1121342112`).  
   - May include special characters such as `/`, `-`, `_`, or `.` (e.g., `INV-1234_5678`).  
   - **DO NOT classify a Vendor Number or PO Number** as an Invoice Number.  
   - **Exclude Numbers starting with `160`**.

### 3. E-Invoice Number:
- **Invoice Number**:  
   - May be **numeric only** (e.g., `288617`) or **alphanumeric** (e.g., `TP0000330`).  
   - Can start with `I`, `IE`, or `INV` followed by numbers (e.g., `I3453423`, `IE7645434`, `INV1121342112`).  
   - May include special characters such as `/`, `-`, `_`, or `.` (e.g., `INV-1234_5678`).  
   - **DO NOT classify a Vendor Number or PO Number** as an Invoice Number.  
   - **Exclude Numbers starting with `160`**.

#### Context-Based Invoice Number Extraction:
3. 1. **Contextual Clues**:  
   When extracting an invoice number, look for common phrases related to invoices, such as:
   - "Invoice number"
   - "Invoice ID"
   - "Invoice ref"
   - "Payment reference"
   - "Clearing"
   - "Bill number"
   - "Reference number"

   These terms often precede or follow the invoice number in the text, making it more likely that nearby numeric strings should be classified as invoice numbers.

3.2. **Flexible Pattern Matching**:  
   While a strict pattern for invoice numbers includes starting with `I`, `IE`, or `INV`, also check for any **long numeric strings** (e.g., 10 digits or more) or **alphanumeric sequences** that don't follow PO or Vendor Number rules. Invoice numbers might not always match the expected prefixes but could still be valid.

3.3. **Cross-reference Context**:  
   If a number appears near terms like "payment," "clearing," or "payment advice," prioritize classifying that number as an invoice number. These terms are often linked to invoices, making the number more likely to be an invoice number.

3.4. **Post-Extraction Validation**:  
   After extracting potential invoice numbers:
   - Check the surrounding context to verify if the number is related to an invoice (e.g., is it preceded by "invoice number" or "payment reference"?).
   - Ensure that the number isn't already classified as a **Vendor Number** or **PO Number**.
   - If a number is in an ambiguous position but follows the invoice-related terms, classify it as an invoice number.

3.5. **Fallback to Long Numeric Sequences**:  
   In the absence of clear context keywords, check if there are any **long numeric sequences** (e.g., 10 digits or more) in the text that could be the invoice number. Use this as a fallback check if no other invoice number is explicitly mentioned.

### Example:
Given the text:
> "Can you please provide payment advice note for this clearing 2168559877 14.08.2023."

The number `2168559877` could be classified as an invoice number because it appears near the term "clearing," which is commonly associated with invoices or payments, even though it doesn't follow the typical invoice number pattern.


### 4. E-Invoice Number:
- **E-Invoice Number**:  
   - Must start with **"E"** and be exactly **10 characters long**.  
   - The E-Invoice Number must be strictly **alphanumeric** (e.g., `E123456789`).  
   - **DO NOT classify any Invoice Number as an E-Invoice Number**.  
   - **E-Invoice Number cannot start with "I"**.

### 5. Other Numbers:
- Any number that **does not follow any of the above rules** (for Vendor, PO, Invoice, or E-Invoice Number) should be classified as **Other Numbers**. 
For example:  
   - Numbers not starting with any valid prefix for Vendor,  PO, or Invoice should be classified as "Other Number."
   - Try to extract most of the numbers/alphanumeric numbers present in the text.
   - **Double check all extracted other numbers**. They could be a potential Vendor, PO, or Invoice number that may have been missed. Thoroughly verify each "Other Number" to make sure it doesn't fit any of the above rules.  
   - If any of these "Other Numbers" matches a valid Vendor Number, PO Number, or Invoice Number (after rechecking), add them in the corresponding final output value.
   - Before adding  it/them make sure that particular numbers is following all the rules specified for that category. Follow the rules properly.

---

Important:  

## Output Format:
Return the extracted information in the following **exact JSON format**:

{{
  "response": {{
    "vendor_name": "<Extracted Vendor Name(s)>",
    "vendor_number": "<Extracted Vendor Number(s)>",
    "po_number": "<Extracted PO Number(s)>",
    "invoice_number": "<Extracted Invoice Number(s)>",
    "e-invoice_number": "<Extracted E-Invoice Number(s)>",
    "other_number": "<Extracted Other Number(s)>"
  }}
}}

---

"""




COMBINED_PROMPT = """

You are a meticulous data reader and extractor with a keen eye for detail and a mastery of different languages. Your goal is to **thoroughly review the provided email draft** text and **extract only specific information** with absolute precision, following the strict rules outlined below.  You will find the input email subject and body at the end.

## Email Structure:
- The email contains details or concerns about a purchase order and may include fields such as **PO number**, **invoice number**, **E-invoice number**, **vendor number**, **vendor name**, and **date**.
- **Extract only the specified details**—vendor name, vendor number, PO number, invoice number, and E-invoice number.
-  Any **other number** that doesn't fit these categories should be classified as "Other number."

---

## Task:
Extract the following details from the **email subject and body**:

1. **Vendor Name** and **Vendor Number**
2. **Purchase Order (PO) Number**
3. **Invoice Number** and **E-Invoice Number**

---

## General Rules:
1. **Order of Extraction**:  
   - First, extract **Vendor Name** and **Vendor Number**.  
   - Then, extract **PO Number**.  
   - Finally, extract **Invoice Number** and **E-Invoice Number**.

2. **Exclusion of Duplicate Numbers**:  
   - A number must be classified **ONLY ONCE** and **must belong to one specific category** (Vendor, PO, or Invoice).  
   - If a number is already classified as **Vendor Number**, **PO Number**, or **Invoice Number**, **DO NOT reclassify it under any other category**.

3. **Uncertainty**:  
   - If you are **unsure** about any number go through the below rules again. If still not sure , **return an empty result**.  
   - Do not make assumptions or classify numbers incorrectly.

4. **Priority**:  
   - **Always prioritize the email subject** to search for the required information.  
   - If not found in the **subject**, then search the **body**.

5. **Language**:
    - If the language of the input text is Serbian, Greek or Bulgarian, then convert the text into English first and then apply below rules.

---

## Extraction Rules:

### 1. Vendor Name and Vendor Number:
- **Vendor Number**:  
   - Must be exactly **10 digits long** and **only numeric**.  
   - **First two digits of any vendor number** will always be either of `40`, `43`, `20`, `30`, `70`, or `90`.  
   - **Exclude Numbers** where first two digits are either of `45`, `47`, `65`,  or '80' . 
   - **Double check email subject** to find the Vendor number. It is mostly present in email subject only.
   - **Look for all such numbers in email subject and body.  Extract all valid Vendor Numbers** found.

- **Vendor Name**:  
   - Must be a **proper firm/company/individual name**.  
   - Vendor Name is typically **preceded or followed by a valid Vendor Number**.  
   - **Do NOT consider names containing invalid patterns**, such as "Coca-Cola" or names with symbols like "-".

### 2. Purchase Order (PO) Number:
- **PO Number**:  
   - Must be exactly **10 digits long** and **numeric only** (may include special characters like `/` such as `45/12345678`).  
   - **Must start with `45`, `47`, or `65`**.  
   - **Exclude numbers with **First two digits** are `40`, `43`, `20`, `30`, `70`, and `90`.  
   - **Look for all such numbers in email subject and body.  Extract all valid PO Numbers** found.

### 3. Invoice Number:
- **Invoice Number**:  
   - Important: First detect the language of the text. If it is in Serbian, Bulgaria or Greek, convert it into english first then apply below rules.
   - May be **numeric only** (e.g., `288617`) or **alphanumeric** (e.g., `TP0000330`).  
   - Can start with `I`, `IE`, or `INV` followed by numbers (e.g., `I3453423`, `IE7645434`, `INV1121342112`).  
   - May include special characters such as `/`, `-`, `_`, or `.` (e.g., `INV-1234_5678`).  
   - **DO NOT classify a Vendor Number or PO Number** as an Invoice Number.  
   - **Exclude Numbers starting with `160`**.

### 3. E-Invoice Number:
- **Invoice Number**:  
   - May be **numeric only** (e.g., `288617`) or **alphanumeric** (e.g., `TP0000330`).  
   - Can start with `I`, `IE`, or `INV` followed by numbers (e.g., `I3453423`, `IE7645434`, `INV1121342112`).  
   - May include special characters such as `/`, `-`, `_`, or `.` (e.g., `INV-1234_5678`).  
   - **DO NOT classify a Vendor Number or PO Number** as an Invoice Number.  
   - **Exclude Numbers starting with `160`**.

#### Context-Based Invoice Number Extraction:
3. 1. **Contextual Clues**:  
   When extracting an invoice number, look for common phrases related to invoices, such as:
   - "Invoice number"
   - "Invoice ID"
   - "Invoice ref"
   - "Payment reference"
   - "Clearing"
   - "Bill number"
   - "Reference number"

   These terms often precede or follow the invoice number in the text, making it more likely that nearby numeric strings should be classified as invoice numbers.
3.2. **Flexible Pattern Matching**:  
   While a strict pattern for invoice numbers includes starting with `I`, `IE`, or `INV`, also check for any **long numeric strings** (e.g., 10 digits or more) or **alphanumeric sequences** that don't follow PO or Vendor Number rules. Invoice numbers might not always match the expected prefixes but could still be valid.

3.3. **Cross-reference Context**:  
   If a number appears near terms like "payment," "clearing," or "payment advice," prioritize classifying that number as an invoice number. These terms are often linked to invoices, making the number more likely to be an invoice number.

3.4. **Post-Extraction Validation**:  
   After extracting potential invoice numbers:
   - Check the surrounding context to verify if the number is related to an invoice (e.g., is it preceded by "invoice number" or "payment reference"?).
   - Ensure that the number isn't already classified as a **Vendor Number** or **PO Number**.
   - If a number is in an ambiguous position but follows the invoice-related terms, classify it as an invoice number.

3.5. **Fallback to Long Numeric Sequences**:  
   In the absence of clear context keywords, check if there are any **long numeric sequences** (e.g., 10 digits or more) in the text that could be the invoice number. Use this as a fallback check if no other invoice number is explicitly mentioned.

### Example:
Given the text:
> "Can you please provide payment advice note for this clearing 2168559877 14.08.2023."

The number `2168559877` could be classified as an invoice number because it appears near the term "clearing," which is commonly associated with invoices or payments, even though it doesn't follow the typical invoice number pattern.


### 4. E-Invoice Number:
- **E-Invoice Number**:  
   - Must start with **"E"** and be exactly **10 characters long**.  
   - The E-Invoice Number must be strictly **alphanumeric** (e.g., `E123456789`).  
   - **DO NOT classify any Invoice Number as an E-Invoice Number**.  
   - **E-Invoice Number cannot start with "I"**.

### 5. Other Numbers:
- Any number that **does not follow any of the above rules** (for Vendor, PO, Invoice, or E-Invoice Number) should be classified as **Other Numbers**. 
For example:  
   - Numbers not starting with any valid prefix for Vendor,  PO, or Invoice should be classified as "Other Number."
   - Try to extract most of the numbers/alphanumeric numbers present in the text.
   - **Double check all extracted other numbers**. They could be a potential Vendor, PO, or Invoice number that may have been missed. Thoroughly verify each "Other Number" to make sure it doesn't fit any of the above rules.  
   - If any of these "Other Numbers" matches a valid Vendor Number, PO Number, or Invoice Number (after rechecking), add them in the corresponding final output value.
   - Before adding  it/them make sure that particular numbers is following all the rules specified for that category. Follow the rules properly.

---

Important:  

## Output Format:
Return the extracted information in the following **exact JSON format**:

{
  "response": {
    "vendor_name": "<Extracted Vendor Name(s)>",
    "vendor_number": "<Extracted Vendor Number(s)>",
    "po_number": "<Extracted PO Number(s)>",
    "invoice_number": "<Extracted Invoice Number(s)>",
    "e-invoice_number": "<Extracted E-Invoice Number(s)>",
    "other_number": "<Extracted Other Number(s)>"
  }
}

---


Below is the subject and body of the email:

{subject}: Subject of the email
{body}: Content/body of email


"""

TEST = """

You are a meticulous data reader and extractor with a keen eye for detail and a mastery of different languages. 
Your goal is to **thoroughly review the provided email subject and body** text and **extract only specific information** with absolute precision, following the strict rules outlined below.  
You will find the input email subject text and body text for analysis at the end.

# Email Structure & Query Types

The emails you receive can be categorized based on their content and purpose. The following are common types of emails that you encounter:

## 1. **Red Letter/Reminder**:
- **Description**: These emails are often sent as reminders for overdue payments, outstanding issues, or missed deadlines. They typically contain a sense of urgency.
- **Content to Extract**: These emails may contain **PO numbers**, **invoice details**, and other **payment-related queries**.

## 2. **PO Query**:
- **Description**: These emails are related to queries or issues regarding **purchase orders**. They may ask for clarification on order details or updates.
- **Content to Extract**: **PO number**, **vendor name**, and possibly details on the status or discrepancies in the PO.

## 3. **Payment Query**:
- **Description**: These emails generally inquire about the status of a **payment**. They may include payment amounts, due dates, or request updates on payment processing.
- **Content to Extract**: **Invoice number**, **payment due date**, **PO number**, and the corresponding **vendor details**.

## 4. **Invoice Query**:
- **Description**: These emails are sent to address questions or issues related to **invoices**. This could include incorrect amounts, missing invoices, or disputes.
- **Content to Extract**: **Invoice number**, **PO number**, **vendor name**, and the vendor's unique identification number.

## 5. **Data Maintenance**:
- **Description**: These emails request updates, corrections, or changes to **data records**, such as **vendor information**, **PO details**, or **invoice records**.
- **Content to Extract**: **Vendor number**, **vendor name**, **PO number**, and any details regarding the required data updates.

## 6. **Other Finance and Non-Finance Query**:
- **Description**: This category includes a variety of queries that do not fall under the other specific categories. These emails may involve **general inquiries** or **non-specific issues** in finance or other business areas.
- **Content to Extract**: This could include any of the specified fields (**vendor number**, **PO number**, etc.), but may also include **other reference numbers** that should be classified as **"Other number"**.

---

## **Details to Extract**

From each email, we are interested in extracting the following **specific details** (which may appear in various formats):

- **PO number**: Unique identifier for the **purchase order**.
- **Invoice number**: Unique identifier for the **invoice** associated with the purchase order.
- **E-invoice number**: Unique identifier for the **electronic version** of the invoice.
- **Vendor number**: Unique identifier for the **vendor**.
- **Vendor name**: Name of the **vendor** providing the goods or services.

## **Classification of Numbers**

Any **number** found in the email that doesn't match the categories above should be classified as **"Other number"**. 
These could be numbers related to **other reference IDs** or unrelated **financial details**.

---

## General Rules:
1. **Order of Extraction**: 
   - First, extract **Vendor Name** and **Vendor Number**.  
   - Then, extract **PO Number**.  
   - Finally, extract **Invoice Number** and **E-Invoice Number**.
   - Extract the remaining numbers and alphanumeric numbers as **Other Number**.
   - Read all **Other numbers** again and double check whether any of those numbers meet the criteria for any of above numbers.


2. **Exclusion of Duplicate Numbers**:
   - A number must be classified **ONLY ONCE** and **must belong to one specific category** (Vendor, PO, or Invoice).  
   - If a number is already classified as **Vendor Number**, **PO Number**, or **Invoice Number**, **DO NOT reclassify it under any other category**.

3. **Uncertainty**:  
   - If you are **unsure** about any number go through the below rules again. If still not sure , **return an empty result**.  
   - Do not make assumptions or classify numbers incorrectly.

4. **Priority**:  
   - **Always prioritize the email subject** to search for the required information.  
   - If not found in the **subject**, then search the **body**.

5. **Language**:
    - If the language of the input text is not English, then convert the text into English first, understand the text well and then apply below rules.
    - Make sure you have detected the language correctly and also translated the text correctly.
---

## Extraction Rules:

### 1. Vendor Name and Vendor Number:
- **Vendor Number**:  
   - Must be exactly **10 digits long** and **only numeric**.  
   - **First two digits of any vendor number** will always be `40`, `43`, `20`, `30`, `70`, and `90`.  
   - **Exclude 10 Digits Numbers** where first two digits are either of `45`, `47`, `65`,  and '80' . 
   - **Double check email subject** to find the Vendor number. It is mostly present in email subject only.
   - **If not found search the email body**.  
   -  **Extract all valid Vendor Numbers** found.

- **Vendor Name**:  
   - Must be a **proper firm/company/individual name**.  
   - Vendor Name is typically **preceded or followed by a valid Vendor Number**.  
   - **Do NOT consider names containing invalid patterns**, such as "Coca-Cola" or names with symbols like "-".

### 2. Purchase Order (PO) Number:
- **PO Number**:  
   - Must be exactly **10 digits long** and **numeric only**.  
   - **Must start with `45`, `47`, or `65`**.  
   - Exclude all 10 digits numbers with **First two digits** are `40`, `43`, `20`, `30`, `70`, and `90`.  
   - If a matching number is preceded by "PO" (like **PO4502702664**) then consider it as a valid PO number.
   - **Look for all such numbers in email subject and body.  Extract all valid PO Numbers** found.

### 3. Invoice Number:
- **Invoice Number**:  
   - Important: First detect the language of the text. If it is not in English, convert it into english first then apply below rules.
   - May be **numeric only** (e.g., `288617`) or **alphanumeric** (e.g., `TP0000330`).  
   - Can start with `I`, `IN`, or `INV` followed by numbers (e.g., `I3453423`, `IN7645434`, `INV-1121`).  
   - May include special characters such as `/`, `-`, `_`, or `.` (e.g., `INV-1234_5678`).  
   - **DO NOT classify a Vendor Number or PO Number** as an Invoice Number.  
   - **Exclude Numbers starting with `160`**.

#### Context-Based Invoice Number Extraction:
3. 1. **Contextual Clues**:  
   When extracting an invoice number, look for common phrases related to invoices, such as:
   - "Invoice number"
   - "Invoice ID"
   - "Invoice ref"
   - "Payment reference"
   - "Clearing"
   - "Bill number"
   - "Reference number"

   These terms often precede or follow the invoice number in the text, making it more likely that nearby numeric strings should be classified as invoice numbers.

3.2. **Flexible Pattern Matching**:  
   While a strict pattern for invoice numbers includes starting with `I`, `IE`, or `INV`, also check for any **long numeric strings** (e.g., 10 digits or more) or **alphanumeric sequences** that don't follow PO or Vendor Number rules. Invoice numbers might not always match the expected prefixes but could still be valid.

3.3. **Cross-reference Context**:  
   If a number appears near terms like "payment," "clearing," or "payment advice," prioritize classifying that number as an invoice number. These terms are often linked to invoices, making the number more likely to be an invoice number.

3.4. **Post-Extraction Validation**:  
   After extracting potential invoice numbers:
   - Check the surrounding context to verify if the number is related to an invoice (e.g., is it preceded by "invoice number" or "payment reference"?).
   - Ensure that the number isn't already classified as a **Vendor Number** or **PO Number**.
   - If a number is in an ambiguous position but follows the invoice-related terms, classify it as an invoice number.

3.5. **Fallback to Long Numeric Sequences**:  
   In the absence of clear context keywords, check if there are any **long numeric sequences** (e.g., 10 digits or more) in the text that could be the invoice number. Use this as a fallback check if no other invoice number is explicitly mentioned.

### Example:
Given the text:
> "Can you please provide payment advice note for this clearing 2168559877 14.08.2023."

The number `2168559877` could be classified as an invoice number because it appears near the term "clearing," which is commonly associated with invoices or payments, even though it doesn't follow the typical invoice number pattern.


### 4. E-Invoice Number:
- **E-Invoice Number**:  
   - Must start with **"E"** and be exactly **10 characters long**.  
   - The E-Invoice Number must be strictly **alphanumeric** (e.g., `E123456789`).  
   - **DO NOT classify any Invoice Number as an E-Invoice Number**.  
   - **E-Invoice Number cannot start with "I"**.

### 5. Other Numbers:
- Any number that **does not follow any of the above rules** (for Vendor, PO, Invoice, or E-Invoice Number) should be classified as **Other Numbers**. 
For example:  
   - Numbers not starting with any valid prefix for Vendor,  PO, or Invoice should be classified as "Other Number."
   - Try to extract most of the numbers/alphanumeric numbers present in the text.

## **Expanded General Rule for  **Handling "Other Numbers"**:

After extracting the specified fields (**Vendor Name**, **Vendor Number**, **PO Number**, **Invoice Number**, and **E-Invoice Number**), follow these steps to handle any remaining numbers:

### **Step 1: Extract Remaining Numbers**:
- Identify all remaining **numbers** or **alphanumeric strings** from the email that are not part of the above categories.
  - These **remaining numbers** should initially be treated as **"Other Numbers"**.

### **Step 2: Double Check for Overlaps**:
- Once the **Other Numbers** are identified, perform a **second check** to verify if any of these numbers could belong to one of the following categories:
  - **PO Number**: Check if the number fits the format of a **Purchase Order** (e.g., numeric sequences of a 10 digit length with first two digits starting with either of 45, 47 or 65).
  - **Invoice Number**: Check if the number looks like an **Invoice Number** (e.g., alphanumeric with a prefix like "INV").
  - **E-Invoice Number**: Ensure the number follows the **E-Invoice format** (e.g., alphanumeric with "EINV" prefix).
  - **Vendor Number**: Verify if it matches the **Vendor Number format** (e.g., numeric sequences of a 10 digit length with first two digits starting with either of 43, 40,  20, 30, 70 or 90).

### **Step 3: Classification as "Other Number"**:
- After thoroughly checking the remaining numbers, if none of them match the known formats or categories for **PO Number**, **Invoice Number**, **E-Invoice Number**, or **Vendor Number**, classify them explicitly as **"Other Number"**.
  - These could include:
    - Reference numbers
    - Internal tracking numbers
    - Unrelated identifiers

### **Example**:
Suppose you've extracted the following numbers from an email:
- **PO Number**: `123456`
- **Invoice Number**: `INV987654`
- **E-Invoice Number**: `EINV123456`
- **Vendor Number**: `4004320032`
- **Other reference number**: `99999`

You would perform a second check to see if `99999` could be:
- A **PO Number**: It doesn't match the expected PO number format.
- An **Invoice Number**: It doesn't fit the alphanumeric format of an invoice number.
- An **E-Invoice Number**: It doesn't follow the "EINV" pattern.
- A **Vendor Number**: It doesn't match the vendor number format.

Since `999999` doesn't match any of these, it is classified as an **Other Number**.

### **Step 4: Final Check for Other Numbers**:
- If you are unsure about whether a number should be categorized as an **Other Number**, look for contextual clues within the email:
  - Is the number **labeled** as a reference number or unrelated to PO/Invoice/Vendor info?
  - Does it appear in a **context** that doesn't relate to the required fields?

- If in doubt, treat it as an **Other Number**, unless it clearly matches one of the specified categories.

---
### Total Token Size Calculation for LLM Request

Please follow the steps below to calculate the total token count for both the input and output:

#### Instructions:
- Calculate the token count for both the **input text** and the **output text**.
- Tokens should include all words, punctuation marks, and special characters. 
- Provide:
  - The number of tokens in the **input text**.
  - The number of tokens in the **output text**.
  - The **total token count** (sum of both).
---
Important:  

## Output Format:
Return the extracted information in the following **exact JSON format**:

{{
  "response": {{
    "vendor_name": "<Extracted Vendor Name(s)>",
    "vendor_number": "<Extracted Vendor Number(s)>",
    "po_number": "<Extracted PO Number(s)>",
    "invoice_number": "<Extracted Invoice Number(s)>",
    "e-invoice_number": "<Extracted E-Invoice Number(s)>",
    "other_number": "<All Extracted Other Number(s)>",
    "token_size": "<Calculated total token count>",
    "other_numbers": {{
      "Extracted Number1": "<Extracted Number title/type/name>",
      "Extracted Number2": "<Extracted Number title/type/name>"
    }}
  }}
}}

---

Below is the subject and body of the email:

{subject}: Subject of the email
{body}: Content/body of email

---
"""

# {{
#   "response": {{
#     "vendor_name": "<Extracted Vendor Name(s)>",
#     "vendor_number": "<Extracted Vendor Number(s)>",
#     "po_number": "<Extracted PO Number(s)>",
#     "invoice_number": "<Extracted Invoice Number(s)>",
#     "e-invoice_number": "<Extracted E-Invoice Number(s)>",
#     "other number": <All Extracted Other Number(s)>",
#     "token_size": "<Calculated total token count>"
#
# }}
#     {{"other_number": {"Extracted Number1": "<Extracted Number title/type/name>",
#                                         "Extracted Number2":" <Extracted Number title/type/name>",
#                                          ...}
#                                          }}
#
# }}
# {{
#   "response": {{
#     "vendor_name": "<Extracted Vendor Name(s)>",
#     "vendor_number": "<Extracted Vendor Number(s)>",
#     "po_number": "<Extracted PO Number(s)>",
#     "invoice_number": "<Extracted Invoice Number(s)>",
#     "e-invoice_number": "<Extracted E-Invoice Number(s)>",
#     "other_number": "<Extracted Other Number(s)>",
#     "total_tokens": <Calculated total token count>
#   }}
# }}

prompt_PO1 = """
You are a financial validator for Purchase Orders (POs) tasked with extracting and validating the PO Number from an invoice. 
Your goal is to extract the relevant PO details and ensure their accuracy by validating them against the **POLineDetails** dataset. 
Perform the necessary validation checks to confirm the integrity of the PO Number.

### Extract the following fields from the invoice:
- **PO Number** - It should be 10 digit number starting with 45 or 47 or 55 or 65 or 67 or 49 or 59.
- **Company Code**
- **Vendor ID**

{inputdata}: Input data is in JSON payload  


Perform the below Scenarios

### **Scenarios:**

(1) #### **If Single PO Number is Extracted**

- **Validate the Extracted PO Number** against the **POLineDetails** dataset using the extracted:
  - **PO Number**
  - **Company Code**
  - **Vendor ID**
Use the below given format to validate 
*Extracted PO Number+Company Code+Vendor ID* 
Then Check with **POLineDetails** and perform the below Conditions
    (1.1) **If There is only Single match Found**:
        - Success - Populate the **PO Number** at the header level. And populate a **PO Line Item** and a **Good Receipt/Service Entry Sheet** on line level.

        - **el-If No Match Found**:
          - **Implement the logic of 'PO Line Item Identification'**.
          Then Perform the below Scenarios:
      
              **If Single Match Found**: Populate the **PO Number** at the header level. And Assign a **PO Line Item** and a **Good Receipt/Service Entry Sheet** on line level.
              **If No Match Found**:Leave PO number empty and add "Invalid or missing PO" reason code.
              **If Multiple Matches Found**:Leave PO number empty and add “Invalid or missing PO” reason code.

    (1.2)**If It Multiple Matches Found**:
        -Leave the **PO Number empty** and add "Invalid or missing PO" reason code.

(2) #### **If PO Number is not Extracted**

 - Implement the Logic of Identification of PO line item. -> 

  And perform the below Scenarios
        **If Single Match Found**
        - Populate the **PO Number** at the header level. And Assign a **PO Line Item** and a **Good Receipt/Service Entry Sheet** on line level.
        **If No Match Found**
        - Leave the PO number Empty
        **If Multiple Matches Found**
        - Leave PO Number Empty and add "Invalid or missing PO" reason code
  

### **Additional Notes:**
- Ensure every scenario is checked and **do not add** any extra content or notes in the output.
- If the PO Number is not validated, ensure the invoice is flagged with the appropriate reason code.

### **Expected Output Format:**
Return the extracted PO validation results in **JSON format**. Use the Input data format.
"""

prompt_PO2 = """
You are a financial validator for Purchase Orders (POs) tasked with extracting and validating the PO Number from an invoice. Your goal is to extract the relevant PO details and ensure their accuracy by validating them against the **POs_ANM** dataset. Perform the necessary validation checks to confirm the integrity of the PO Number.

### Extract the following fields from the invoice:
- **PO Number** - It should be 10 digit number starting with 45 or 47 or 55 or 65 or 67 or 49 or 59.
- **Company Code**
- **Vendor ID**

{inputdata}: Input data is in JSON payload

The Scenarios are given below, Check which Scenario is Perfectly match
### **Scenarios:**

First Scenario(1) #### **If *Single* PO Number is Extracted from Invoice **Peform the below operations else skip to Second Scenario(2).

- **Validate the Extracted PO Number** against the **POLineDetails** dataset using the extracted:
  - **PO Number**
  - **Company Code**
  - **Vendor ID**

Then Analysis every rule and perform the rule which is suited to extracted PO Number by the below given rules(1,2,3) that which is perfectly match with **POLineDetails** Dataset.
    Rule(1) **If extracted PO Number is matches with *Only* One PO Number against check in Dataset(POLineDetails) Then perform the below operation else skip to check the Rule(2).

        - Success - Populate the **PO Number** at the header level. And then implement the **logic of 'PO Line Item Identification** to populate the **POLine Number** and **Gr/SES**
        **Logic of 'PO Line Item Identification'** 
        Compare the below given fields in LineDetails with POLineDetails to find which matches perfectly 
        ‘lineDescription’ field  against extracted line item/s description from invoice​
        ‘unitprice’ field  against extracted line item/s unit price from invoice​
        ‘LineQuantity’ field against extracted line item/s quantity from invoice​
        ‘deliveryNote’ field against extracted invoice number or delivery note number from invoice
        
        And populate a **PO Line Item** and a **Good Receipt/Service Entry Sheet** on line level for every line Details.

    Rule(2)**el-if No Match Found**:
          - **Implement the logic of 'PO Line Item Identification'**.
          Then Perform the below Scenarios:
      
              **If Single Match Found**: Populate the **PO Number** at the header level. And Assign a **PO Line Item** and a **Good Receipt/Service Entry Sheet** on line level.
              **If No Match Found**:Leave PO number empty and add "Invalid or missing PO" reason code.
              **If Multiple Matches Found**:Leave PO number empty and add “Invalid or missing PO” reason code.

    Rule(3)**el-if extracted PO Number is matches with *Many* PO Number against check in Dataset(POLineDetails) Then perform the below operation else skip to check the next Scenario.
        -Leave the **PO Number empty** and add "Invalid or missing PO" reason code.

Second Scenario(2) #### **If PO Number is not Extracted**

 - Implement the Logic of Identification of PO line item. -> 

  And perform the below Scenarios
        **If Single Match Found**
        - Populate the **PO Number** at the header level. And Assign a **PO Line Item** and a **Good Receipt/Service Entry Sheet** on line level.
        **If No Match Found**
        - Leave the PO number Empty
        **If Multiple Matches Found**
        - Leave PO Number Empty and add "Invalid or missing PO" reason code
  

### **Additional Notes:**
- Ensure every scenario is checked and **do not add** any extra content or notes in the output.
- If the PO Number is not validated, ensure the invoice is flagged with the appropriate reason code.

### **Expected Output Format:**
Return the extracted PO validation results in **JSON format**. Use the Input data format.
"""


prompt_POnew="""
## **Role & Objective**  
You are an **expert in data processing and intelligent text matching**.  
Your task is to process a JSON object containing invoice data, accurately match descriptions, quantities, unit prices, and invoice numbers between `"LineDetails"` and `"POLineDetails"`, and populate missing fields in `"LineDetails"` based on specific conditions while maintaining data integrity.  

---

{inputdata}: Input data is in JSON payload

## **Instructions**

### **1. Understand the Data Structure**  
The input JSON contains three main sections:  
1. **`"Header"`** – Contains general invoice details, including `"PO Number"`.  
2. **`"LineDetails"`** – Lists individual line items from the invoice, each with `"Description"`, `"Quantity"`, and `"Unit Price"`.  
3. **`"POLineDetails"`** – Lists purchase order details, each with `"lineDescription"`, `"lineQuantity"`, `"unitPrice"`, and `"deliveryNote"`.  

Your job is to:  
- Extract relevant details from these sections.  
- Match `"Description"` from `"LineDetails"` with `"lineDescription"` from `"POLineDetails"`.  
- Match `"Quantity"` from `"LineDetails"` with `"lineQuantity"` from `"POLineDetails"`.  
- Match `"Unit Price"` from `"LineDetails"` with `"unitPrice"` from `"POLineDetails"`.  
- Match `"Delivery note/Invoice Number"` from `"Header"` with `"deliveryNote"` from `"POLineDetails"`.  
- Populate missing fields in `"LineDetails"` based on the best possible matches.
*Once the Missing fields values are populated then proceed to populate the '"PO Number" in "Header" only from '"LineDetails"'.
- If `"PO Number"` is missing or empty in `"Header"`, populate it **only from PO Numbers found in `"LineDetails"`**. Do not take the PO Number from `"POLineDetails"`.

**Strict Instructions:**
- **Populate missing values from `"POLineDetails"` only if two or more of the following fields match**:  
  1. `"Description"` from `"LineDetails"` vs. `"lineDescription"` from `"POLineDetails"`  
  2. `"Quantity"` from `"LineDetails"` vs. `"lineQuantity"` from `"POLineDetails"`  
  3. `"Unit Price"` from `"LineDetails"` vs. `"unitPrice"` from `"POLineDetails"`  
  4. `"Delivery note/Invoice Number"` from `"Header"` vs. `"deliveryNote"` from `"POLineDetails"`  

- If only **one** field matches, **do not populate anything**.  
- **The `"PO Number"` should only be populated from PO Numbers found in `"LineDetails"`. It should **not** be populated from `"POLineDetails"`.**

---

### **2. Extract PO Number**  
- Locate the `"PO Number"` from the `"Header"` section.  
- If `"PO Number"` is missing or empty, populate it in `"Header"` with all the **PO Numbers** found in `"LineDetails"`.  
- **DO NOT take the `"PO Number"` from `"POLineDetails"`.** Only populate the PO Number from `"LineDetails"`.

---

### **3. Extract Descriptions, Quantities, and Prices**  
- Retrieve `"Description"` from each entry in `"LineDetails"`.  
- Retrieve `"lineDescription"` from each entry in `"POLineDetails"`.  
- Retrieve `"Quantity"` from each entry in `"LineDetails"`.  
- Retrieve `"lineQuantity"` from each entry in `"POLineDetails"`.  
- Retrieve `"Unit Price"` from each entry in `"LineDetails"`.  
- Retrieve `"unitPrice"` from each entry in `"POLineDetails"`.  
---

### **4. Match Fields**  
- **Description Matching**: Compare `"Description"` from `"LineDetails"` with `"lineDescription"` from `"POLineDetails"`.  
  - Use **fuzzy matching** to handle:
    - Abbreviations  
    - Minor spelling variations  
    - Differences in phrasing  
  - A match is valid if the descriptions are **highly similar** (share key terms or are minor variations of each other).

- **Quantity Matching**: Compare `"Quantity"` from `"LineDetails"` with `"lineQuantity"` from `"POLineDetails"`.  
  - The quantities must match **exactly** (no tolerance for discrepancies).

- **Unit Price Matching**: Compare `"Unit Price"` from `"LineDetails"` with `"unitPrice"` from `"POLineDetails"`.  
  - The unit prices must match **exactly** (no tolerance for discrepancies).

- **Invoice Number Matching**: Compare `"Delivery note/Invoice Number"` from `"Header"` with `"deliveryNote"` from `"POLineDetails"`.  
  - The invoice numbers must match **exactly** (no tolerance for discrepancies).

---

### **5. Populate Missing Data**  
For each **matched** `"Description"` and `"lineDescription"`, update `"LineDetails"` as follows:

| **`LineDetails` Field**        | **Value to Populate From `POLineDetails`** |
|--------------------------------|-------------------------------------------|
| `"PO Line Number"`             | `"erpLineNumber"`                         |
| `"GR/ Service Entry Sheet"`    | `"erpGRNumber"`                           |
| `"Purchase Order"`             | `"PoNumber"`                              |

#### **Important Notes:**  
- **Populate the missing fields in `"LineDetails"` only if two or more of the following fields match**:
  1. `"Description"` from `"LineDetails"` vs. `"lineDescription"` from `"POLineDetails"`.
  2. `"Quantity"` from `"LineDetails"` vs. `"lineQuantity"` from `"POLineDetails"`.
  3. `"Unit Price"` from `"LineDetails"` vs. `"unitPrice"` from `"POLineDetails"`.
  4. `"Delivery note/Invoice Number"` from `"Header"` vs. `"deliveryNote"` from `"POLineDetails"`.
  
- **If only one field matches**, **do not populate anything**.
- **The `"PO Number"` should be populated only from PO Numbers found in `"LineDetails"`. Do not use `"PO Number"` from `"POLineDetails"`.**

---

### **6. Handle Edge Cases**  
You must account for the following scenarios:  
- If `"LineDetails"` is **empty**, return the original JSON without modifications.  
- If `"POLineDetails"` is **empty**, return `"LineDetails"` without additional values.   
- If `"Header"` is empty or missing, process only `"LineDetails"`.  

---

### **7. Output Format**  
- The final output must be a **structured JSON object** that retains all original values from `"Header"` and `"LineDetails"`, while incorporating updated values where matches exist.  
- All keys should remain present, even if their values are empty.  
- Ensure the JSON is **well-formatted and valid**.

---

## **Final Deliverable**  
- A **clean, structured JSON** with the best possible matches populated.  
- No incorrect data should be introduced.  
- The format must be maintained exactly as the input JSON, with additional populated values where applicable.
"""

prompt_POGood= """
###**Role & Objective**
You are an expert in data processing and intelligent text matching. Your task is to process a structured JSON object containing invoice data, accurately match descriptions between "LineDetails" and "POLineDetails", and populate missing fields while maintaining data integrity.
You must ensure that all scenarios are handled correctly and that the final output remains structured, complete, and accurate.
Follow the instructions carefully to extract, match, and update data while preserving the original structure.

##**Input JSON Format**
The input JSON consists of three main sections:

"Header": Contains invoice metadata, including "PO Number", "Invoice Number", and financial details.
"LineDetails": Contains multiple line items with descriptions, amounts, and order references.
"POLineDetails": Contains multiple purchase order details, including "lineDescription" and "erpLineNumber".

Below is the input JSON:

{inputdata}

###**Processing Instructions**

##**Step 1: Extract PO Number**
Locate the "PO Number" field from the "Header" section.
If "PO Number" is missing or empty, proceed without it.

##**Step 2: Extract Descriptions, Quantities, Unit Prices, and Invoice Numbers**
Retrieve "Description" from each entry in "LineDetails".
Retrieve "lineDescription" from each entry in "POLineDetails".
Retrieve "Quantity" from each entry in "LineDetails".
Retrieve "lineQuantity" from each entry in "POLineDetails".
Retrieve "Unit Price" from each entry in "LineDetails".
Retrieve "unitPrice" from each entry in "POLineDetails".
Retrieve "Delivery Note" or "Invoice Number" from "Header".
Retrieve "deliveryNote" from each entry in "POLineDetails".

### **Step 3: Match PO Line Items**
*How to Identify Matching PO Lines*:
A match occurs when **at least two** of the following fields in the "LineDetails" and the "POLineDetails" are exactly the same:

Invoice Line Field (LineDetails)	PO Line Field (POLineDetails)
"Description"	"lineDescription"
"Unit Price"	"unitPrice"
"Quantity"	"lineQuantity"
"Header.Delivery Note" / "Header.Invoice Number"	"deliveryNote"

**Rules for Matching**:
'"Match"': If **two or more** fields match between the "LineDetails" and the "POLineDetails". Populate the missing values in the "LineDetails" with data from the matching "POLineDetails".
'"Non-Match"': If **only one** field matches between the "LineDetails" and the "POLineDetails". *Do not* populate any values in the "LineDetails".Do not consider it a match—leave the Invoice Line unchanged.

##**Key Points**:
Populate the "LineDetails" only if **two or more** fields match.
Do not populate the "LineDetails" if **only one** field matches.
Ensure that Non-Match lines are *not* populated at all.

##**Step 4: Populate Missing Data**
For each matched "LineDetails" and "POLineDetails", update "LineDetails" as follows:

LineDetails Field	Value to Populate From POLineDetails
"Purchase Order"	"PoNumber"
"PO Line"	"erpLineNumber"
"GR/ Service Entry Sheet"	"erpGRNumber"

##**Step 5: Populate PO Number on Invoice Header**

- Populate *all* assigned PO Numbers found in "LineDetails" *Only* into the Header "PO Number".
- Keep the "Header.PO number" unchanged irrespective of PO number found in "LineDetails".
- If Any new PO number found in "LineDetails", it should be *appended* to the "Header.PO Number", separated by a comma (',').
- If multiple PO Numbers exist, they should be stored as a comma-separated array in "Header.PO Number".
- Ensure that the "Header.PO Number" is populated *ONLY* from the PO numbers in "LineDetails", *NOT* from "POLineDetails".
- If no PO number is found in "LineDetails", leave the "PO Number" in "Header" as empty.*Do not* populate the PO Numbers from "POLine details"

### **Important Adjustment**:  
**PO Number Handling Rule**:
- **Do not populate the "Header.PO Number" from "POLineDetails" at any case**. 
- **Populate "Header.PO Number" Only from "LineDetails"**.
- Keep the "Header.PO number" unchanged irrespective of PO number found in "LineDetails".
- If **new PO numbers** are found in "LineDetails", append them to the existing "Header.PO Number" (separated by commas), **without replacing any existing values**.
- If **no PO number** is found in "LineDetails", leave the "Header.PO Number" empty.
  
**Example**:  
"PO Number": "12345, 67890"

##**Step 6: Handle Edge Cases**
You must account for the following scenarios:

If "LineDetails" is empty, return the original JSON without modifications.
If "POLineDetails" is empty, return "LineDetails" without additional values.
If "Header" is empty or missing, process only "LineDetails" and "POLineDetails".
*Do not* populate the PO Number from "POLineDetails". Always populate PO Numbers from "LineDetails" only.
Keep the "Header.PO number" unchanged irrespective of PO number found in "LineDetails".

##**Step 7: Output Format**
Please return the output in STRICT JSON format.

⚠️ Important Requirements:
ONLY output JSON. No explanations, no notes, no additional text.
The JSON must be well-formatted and valid.
All keys must be present, even if their values are empty.
Do not include any extra text like "Step 7: Output Format" or markdown syntax.
Ensure no invalid characters or encoding issues are present.


### JSON Structure:
{{
  "Header": {{
    "Invoice Number": "",
    "Delivery Note": "",
    "PO Number": "",
    "Company Code": "",
    "Vendor ID": "",
    "Tax Rate": "",
    "Invoice Subtotal": "",
    "Header Discount Amount": "",
    "Freight Charges": "",
    "Miscellaneous Charges": "",
    "Tax Amount": "",
    "Withholding Tax Amount": "",
    "Invoice Total": ""
  }},
  "LineDetails": [
    {{
      "LineItemNumber": "",
      "Description": "",
      "Quantity": "",
      "Unit Price": "",
      "Unit of Measurement": "",
      "Amount": "",
      "Material": "",
      "Discount": "",
      "Tax Amount": "",
      "Purchase Order": "",
      "PO Line Number": "",
      "GR/ Service Entry Sheet Number": ""
    }}
  ],
}}
"""

prompt_POcrt="""
###**Role & Objective**
You are an expert in data processing and intelligent text matching. Your task is to process a structured JSON object containing invoice data, accurately match descriptions between "LineDetails" and "POLineDetails", and populate missing fields while maintaining data integrity.
You must ensure that all scenarios are handled correctly and that the final output remains structured, complete, and accurate.
Follow the instructions carefully to extract, match, and update data while preserving the original structure.

**Input JSON Format**
The input JSON consists of three main sections:

"Header": Contains invoice metadata, including "PO Number", "Invoice Number", and financial details.
"LineDetails": Contains multiple line items with descriptions, amounts, and order references.
"POLineDetails": Contains multiple purchase order details, including "lineDescription" and "erpLineNumber".
Below is the input JSON:

{inputdata}

##**Processing Instructions**

**Step 1: Extract PO Number**
Locate the "PO Number" field from the "Header" section.
If "PO Number" is missing or empty, proceed without it.

**Step 2: Extract Descriptions, Quantities, Unit Prices, and Invoice Numbers**
Retrieve "Description" from each entry in "LineDetails".
Retrieve "lineDescription" from each entry in "POLineDetails".
Retrieve "Quantity" from each entry in "LineDetails".
Retrieve "lineQuantity" from each entry in "POLineDetails".
Retrieve "Unit Price" from each entry in "LineDetails".
Retrieve "unitPrice" from each entry in "POLineDetails".
Retrieve "Delivery Note" or "Invoice Number" from "Header".
Retrieve "deliveryNote" from each entry in "POLineDetails".

**Step 3: Match PO Line Items**
**How to Identify Matching PO Lines**:
A match occurs when *at least two* of the following fields in the "LineDetails" and the "POLineDetails" are highly similar:

Invoice Line Field (LineDetails)	PO Line Field (POLineDetails)
"Description"	"lineDescription"
"Unit Price"	"unitPrice"
"Quantity"	"lineQuantity"
"Header.Delivery Note" / "Header.Invoice Number"	"deliveryNote"

**Matching Approach for Each Field**
1. Description Matching (Fuzzy Matching)
The "Description" field in "LineDetails" should be matched against the "lineDescription" field in "POLineDetails" with the following considerations:

Abbreviations: If the description includes abbreviations (e.g., "PC" vs. "Personal Computer"), it should still be considered a match if the overall meaning is the same.
Spelling Variations: Minor spelling differences (e.g., "Laptop" vs. "Laptap", "computer" vs. "computor") should be accounted for.
Phrasing Differences: Variations in phrasing (e.g., "Wireless Mouse" vs. "Mouse Wireless") should be matched if the key terms are consistent.
Plurality: Differences in singular/plural forms (e.g., "Mouse" vs. "Mice") should be considered as a match.
Case Insensitivity: Ensure case-insensitive matching (e.g., "LAPTOP" vs. "Laptop").

2. Unit Price Matching (Exact Match)
For the "Unit Price" field, an exact match between "Unit Price" in "LineDetails" and "unitPrice" in "POLineDetails" is required. No fuzzy matching should be applied here. The values must match exactly (including decimals).

3. Quantity Matching (Exact Match)
The "Quantity" field should be matched exactly between "LineDetails" and "POLineDetails". No fuzzy matching should be applied here. The quantities must be exactly the same in both fields.

4. Delivery Note Matching (Fuzzy Matching)
The "Header.Delivery Note" or "Header.Invoice Number" should be matched exactly with "deliveryNote" in "POLineDetails". 

##**Rules for Matching**:
"Match": If *any two or more* fields match between the "LineDetails" and the "POLineDetails", populate the missing values in "LineDetails" with data from the matching "POLineDetails".
"Non-Match": If *only one* field matches between the "LineDetails" and the "POLineDetails", do not populate any values in the "LineDetails". Leave the Invoice Line unchanged.

##**Step 4: Populate Missing Data**
For each matched "LineDetails" and "POLineDetails", update "LineDetails" as follows:

LineDetails Field	Value to Populate From POLineDetails
"Purchase Order"	"PoNumber"
"PO Line"	"erpLineNumber"
"GR/ Service Entry Sheet"	"erpGRNumber"

##**Step 5: Populate PO Number on Invoice Header**
- Populate all assigned PO Numbers found in "LineDetails" Only into the "Header.PO Number".
- Keep the "Header.PO Number" unchanged irrespective of PO number found in "LineDetails".
- If any new PO number is found in "LineDetails", it should be appended to the "Header.PO Number", separated by a comma (,).
- If multiple PO Numbers exist, they should be stored as a comma-separated array in "Header.PO Number".
- Ensure that the "Header.PO Number" is populated *ONLY* from the PO numbers in "LineDetails", *NOT* from "POLineDetails".
- If no PO number is found in "LineDetails", leave the "Header.PO Number" as *empty*.

##**PO Number Handling Rule**:
Do not populate the "Header.PO Number" from "POLineDetails".
Only populate "Header.PO Number" from "LineDetails".
If new PO numbers are found in "LineDetails", append them to the existing "Header.PO Number", without replacing any existing values.
If no PO number is found in "LineDetails", leave the "Header.PO Number" empty.

##**Step 6: Handle Edge Cases**
You must account for the following scenarios:

If "LineDetails" is empty, return the original JSON without modifications.
If "POLineDetails" is empty, return "LineDetails" without additional values.
If "Header" is empty or missing, process only "LineDetails" and "POLineDetails".
Do not populate the PO Number from "POLineDetails". Always populate PO Numbers from "LineDetails" only.
Do not remove the existing PO Numbers on the header at any case.
Step 7: Output Format
Please return the output in STRICT JSON format.

⚠️ Important Requirements:

ONLY output JSON. No explanations, no notes, no additional text.
The JSON must be well-formatted and valid.
All keys must be present, even if their values are empty.
Do not include any extra text like "Step 7: Output Format" or markdown syntax.
Ensure no invalid characters or encoding issues are present.

### JSON Structure:
{{
  "Header": {{
    "Invoice Number": "",
    "Delivery Note": "",
    "PO Number": "",
    "Company Code": "",
    "Vendor ID": "",
    "Tax Rate": "",
    "Invoice Subtotal": "",
    "Header Discount Amount": "",
    "Freight Charges": "",
    "Miscellaneous Charges": "",
    "Tax Amount": "",
    "Withholding Tax Amount": "",
    "Invoice Total": ""
  }},
  "LineDetails": [
    {{
      "LineItemNumber": "",
      "Description": "",
      "Quantity": "",
      "Unit Price": "",
      "Unit of Measurement": "",
      "Amount": "",
      "Material": "",
      "Discount": "",
      "Tax Amount": "",
      "Purchase Order": "",
      "PO Line Number": "",
      "GR/ Service Entry Sheet Number": ""
    }}
  ],
}}
"""

prompt_POcrt = """
### **Role & Objective**
You are an expert in data processing and intelligent text matching. Your task is to process a structured JSON object containing invoice data. Your main goal is to accurately match descriptions between "LineDetails" and "POLineDetails" while maintaining data integrity and ensuring the "PO Number" is populated correctly from the "LineDetails".
 
You must handle all scenarios correctly, ensuring the final output is structured, complete, and accurate. Follow the instructions carefully to extract, match, and update data while preserving the original structure.
 
---
 
## **Input JSON Format**
The input JSON consists of three main sections:
1. **Header**: Contains invoice metadata, including "PO Number", "Invoice Number", and financial details.
2. **LineDetails**: Contains multiple line items with descriptions, amounts, and order references.
3. **POLineDetails**: Contains multiple purchase order details, including "lineDescription" and "erpLineNumber".
 
---
 
Below is the input JSON:
 
{inputdata}
 
---
 
### **Processing Instructions**
 
## **Step 1: Extract PO Number from Header**
- Extract the "PO Number" from the **Header** section.
  - If the "PO Number" in the Header is empty, leave it unchanged.
 
---
 
## **Step 2: Extract Data from LineDetails and POLineDetails**
Retrieve the following fields:
- From **LineDetails**:
  - `"Description"`
  - `"Quantity"`
  - `"Unit Price"`
 
- From **POLineDetails**:
  - `"lineDescription"`
  - `"lineQuantity"`
  - `"unitPrice"`
  - `"deliveryNote"` (Invoice Number) from '"Header"'
  - `"erpLineNumber"`
  - `"PoNumber"`
  - `"erpGRNumber"`
 
---
 
## **Step 3: Match PO Line Items**
### **Improved Matching Logic**
A **"POLineDetail" is considered a match** if **at least two** of the following fields match **exactly** between "LineDetails" and "POLineDetails":
 
**Matching Fields:**
1. `"Description"` matches `"lineDescription"` **(Allow minor formatting differences but ensure semantic similarity)**
2. `"Quantity"` matches `"lineQuantity"` **(Ensure numeric equivalence, allowing small rounding differences)**
3. `"Unit Price"` matches `"unitPrice"` **(Ensure precision up to 2 decimal places)**
4. `"Header.Invoice Number"` or `"Header.Delivery Note"` matches `"deliveryNote"`  
   **(Ignore "INV" prefix when comparing "deliveryNote". Example: "8822872" matches "INV8822872")**
 
**If only one field matches, do NOT populate data from POLineDetails. Ensure at least two fields match to establish a valid association.**
 
---
 
### **Step 4: Apply Matching & Populate Data**
For every **matched** "POLineDetails" entry:
- Populate the following fields in **LineDetails**:
  - `"Purchase Order"`: Use `"PoNumber"` from the matched "POLineDetails".
  - `"PO Line Number"`: Use `"erpLineNumber"` from the matched "POLineDetails".
  - `"GR/ Service Entry Sheet Number"`: Use `"erpGRNumber"` from the matched "POLineDetails".
 
**Avoid incorrect mappings:**
- **Ensure a unique match per LineDetails entry.**  
- **If multiple POLineDetails match, prioritize the one with the most fields matching.**  
- **Do not assign multiple POLineDetails to a single LineDetails entry.**  
 
---
 
### **Step 5: Update `Header.PO Number`**
- Always populate the `Header.PO Number` **ONLY from the "LineDetails" section**.
  - If `Header.PO Number` is empty and **LineDetails** contains a valid PO number, populate it.
  - If `Header.PO Number` already has one or more PO numbers, **append any new PO numbers found in "LineDetails"** to the existing PO numbers in `Header.PO Number`, separating them by a comma (`,`). Do **not remove or modify existing PO numbers** in the Header.
 
  **Important**:
  - **Do not** overwrite or modify the existing PO number in `Header.PO Number` if the PO number is already present.
  - If **LineDetails** contains new PO numbers that are **not already in** `Header.PO Number`, append them.
  - Keep the "Header.PO Number" unchanged irrespective of PO number found in "LineDetails".
 
---
 
### **Step 6: Edge Case Handling**
Consider the following scenarios:
- If `LineDetails` is empty, **return the original JSON without modifications**.
- If `POLineDetails` is empty, **return only the "LineDetails" as is**, without any additional values.
- If **no match** is found between "LineDetails" and "POLineDetails", do **not populate any data**.
- If **Header** is empty or missing, **only process "LineDetails"** and "POLineDetails". Leave the `Header.PO Number` unchanged.
- Always **preserve** existing values in `Header.PO Number` and **append** any new PO numbers from "LineDetails", but **do not overwrite** the existing PO numbers in the Header.
 
---
 
### **Step 7: Output Format**
Return the result **strictly** in JSON format. The structure should be maintained with all keys, even if their values are empty.
 
### JSON Structure:
{{
  "Header": {{
    "Invoice Number": "",
    "Delivery Note": "",
    "PO Number": "",
    "Company Code": "",
    "Vendor ID": "",
    "Tax Rate": "",
    "Invoice Subtotal": "",
    "Header Discount Amount": "",
    "Freight Charges": "",
    "Miscellaneous Charges": "",
    "Tax Amount": "",
    "Withholding Tax Amount": "",
    "Invoice Total": ""
  }},
  "LineDetails": [
    {{
      "LineItemNumber": "",
      "Description": "",
      "Quantity": "",
      "Unit Price": "",
      "Unit of Measurement": "",
      "Amount": "",
      "Material": "",
      "Discount": "",
      "Tax Amount": "",
      "Purchase Order": "",
      "PO Line Number": "",
      "GR/ Service Entry Sheet Number": ""
    }}
  ]
}}
 
⚠️ **Important Requirements**:
- **ONLY output JSON**. **No explanations, no notes, no additional text.**
- Ensure the output is **valid JSON** without errors or extra characters.
- All keys must be present in the output, even if they are empty.
- **Ignore case sensitivity when matching fields.**
- Do not include any markdown syntax or additional text.
 
"""
  
prompt_PO = """
### **Role & Objective**
You are an expert in data processing and intelligent text matching. Your task is to process a structured JSON object containing invoice data.
Your primary goal is to accurately populate "Purchase Order", "PO Line Number", and "GR/ Service Entry Sheet Number" in "LineDetails" only when a valid match is found based on strict matching criteria and ensuring the "PO Number" is populated correctly from the "LineDetails". 
You must handle all scenarios correctly, ensuring the final output is structured, complete, and accurate. Follow the instructions carefully to extract, match, and update data while preserving the original structure.
 
---

**Critical Business Rules:**  
1. `"Purchase Order"` in `"LineDetails"` must be populated **ONLY if a matching `"POLineDetails"` is found**.  
   - If **no match** is found, `"Purchase Order"` in `"LineDetails"` **must remain unchanged**.  
2. `"Header.PO Number"` must be derived **ONLY from `"Purchase Order"` values in `"LineDetails"`**.  
   - **Do NOT** use `"PoNumber"` from `"POLineDetails"` to update `"Header.PO Number"`.
3. Existing PO numbers in the header must not be overwritten.

---

## **Input JSON Format**
The input JSON consists of three main sections:
1. **Header**: Contains invoice metadata, including "PO Number", "Invoice Number", and financial details.
2. **LineDetails**: Contains multiple line items with descriptions, amounts, and order references.
3. **POLineDetails**: Contains multiple purchase order details, including "lineDescription" and "erpLineNumber".
 
---
 
Below is the input JSON:
 
{inputdata}
 
---
 
### **Processing Instructions**
 
## **Step 1: Extract PO Number from Header**
- Extract the "PO Number" from the **Header** section.
  - If the "PO Number" in the Header is empty, leave it unchanged.
 
---
 
## **Step 2: Extract Data from LineDetails and POLineDetails**
Retrieve the following fields:
- From **LineDetails**:
  - `"Description"`
  - `"Quantity"`
  - `"Unit Price"`
  - `"Purchase Order"` (**This will be used for `Header.PO Number`**)
 
- From **POLineDetails**:
  - `"lineDescription"`
  - `"lineQuantity"`
  - `"unitPrice"`
  - `"erpLineNumber"`
  - `"PoNumber"` (**This will be used for `"Purchase Order"` in `"LineDetails"` if a match is found**)
  - `"erpGRNumber"`
 
- From **Header** :
  - `"deliveryNote"` and  '"Invoice Note"'
 
---

##**Step 3: Match PO Line Items**
*Logic for PO Line Item Identification:*
A line is considered a *match* if two or more of the following fields match:
Invoice Line Field (LineDetails)	PO Line Field (POLineDetails)
"Description"	"lineDescription"
"Unit Price"	"unitPrice"
"Quantity"	"lineQuantity"
"Header.Delivery Note" / "Header.Invoice Number"	"deliveryNote" (ignore the "INV" prefix when comparing).
A line is considered a *Non-Match* if only one field matches. Do not populate any data(Including PO Number) for non-matching lines.

---
 
### **Step 6: Update LineDetails (Only for Matched Entries)**
For every **matched** "POLineDetails" entry:
- Populate the following fields in **LineDetails**:
  - `"Purchase Order"`: Use `"PoNumber"` from the matched "POLineDetails".
  - `"PO Line Number"`: Use `"erpLineNumber"` from the matched "POLineDetails".
  - `"GR/ Service Entry Sheet Number"`: Use `"erpGRNumber"` from the matched "POLineDetails".

| **LineDetails Field**                     | **Populate From POLineDetails**  |  
|-------------------------------------------|----------------------------------|  
| **"Purchase Order"**                      | `"PoNumber"`                     |  
| **"PO Line Number"**                      | `"erpLineNumber"`                |  
| **"GR/ Service Entry Sheet Number"**      | `"erpGRNumber"`                  |  

 
**Avoid incorrect mappings:**
- **Ensure a unique match per LineDetails entry.**  
- **If multiple POLineDetails match, prioritize the one with the most fields matching.**  
- **Do not assign multiple POLineDetails to a single LineDetails entry.**  
 
---
 
### **Step 7: Update Header.PO Number (Improved Logic)**:

*Strict Rule*: '"Header.PO Number"' must be derived ONLY from "Purchase Order" in LineDetails.

Extract all unique "Purchase Order" numbers from "LineDetails" and populate it into '"Header.Po Number"'.
If `"Header.PO Number"` is empty, populate it with **all  "Purchase Order" numbers** found in `"LineDetails"` as a **comma-separated list**, including multiple values if present ensuring no duplicates.
If "Header.PO Number" is already present:
   - Retain the existing '"Header.PO numbers"'.
   - Append only new "Purchase Order" numbers from "LineDetails", Ensure no Same PO number is repeating.
Do NOT overwrite existing PO numbers in "Header.PO Number" under any condition.
Do NOT use "PoNumber" from "POLineDetails" to update "Header.PO Number".
Ensure the final "Header.PO Number" includes all unique Purchase Order numbers found in "LineDetails", separated by commas if there are multiple values.


**Example Scenario**:

If Header.PO Number = "" and LineDetails has PO numbers [1001, 1002, 1003] → Output: "1001, 1002, 1003"
If Header.PO Number = "1001" and LineDetails has PO numbers [1001, 1002] → Output: "1001, 1002"
If Header.PO Number = "1001, 1003" and LineDetails has PO numbers [1002, 1003] → Output: "1001, 1003, 1002"

---
 
### **Step 8: Edge Case Handling**
Consider the following scenarios:
- If `LineDetails` is empty, **return the original JSON without modifications**.
- If `POLineDetails` is empty, **return only the "LineDetails" as is**, without any additional values.
- If **no match** is found between "LineDetails" and "POLineDetails", do **not populate any data**.
- If **Header** is empty or missing, **only process "LineDetails"** and "POLineDetails". Leave the `Header.PO Number` unchanged.
- Always **preserve** existing values in `Header.PO Number` and **append** any new Purchase Order numbers from "LineDetails", but **do not overwrite** the existing PO numbers in the Header.
 
---
 
### **Step 9: Output Format**
Return the result **strictly** in JSON format. The structure should be maintained with all keys, even if their values are empty.
 
### JSON Structure:
{{
  "Header": {{
    "Invoice Number": "",
    "Delivery Note": "",
    "PO Number": "",
    "Company Code": "",
    "Vendor ID": "",
    "Tax Rate": "",
    "Invoice Subtotal": "",
    "Header Discount Amount": "",
    "Freight Charges": "",
    "Miscellaneous Charges": "",
    "Tax Amount": "",
    "Withholding Tax Amount": "",
    "Invoice Total": ""
  }},
  "LineDetails": [
    {{
      "LineItemNumber": "",
      "Description": "",
      "Quantity": "",
      "Unit Price": "",
      "Unit of Measurement": "",
      "Amount": "",
      "Material": "",
      "Discount": "",
      "Tax Amount": "",
      "Purchase Order": "",
      "PO Line Number": "",
      "GR/ Service Entry Sheet Number": ""
    }}
  ],
  "POLineDetails": [
			[
				{{
					"PoNumber": "",
					"lineDescription": "",
					"lineQuantity": "",
					"unitPrice": "",
					"erpLineNumber": "",
					"unitOfMeasurement": ""
				}}
         ]
      ]          
}}
 
 **Important Requirements**:
- **ONLY output JSON**. **No explanations, no notes, no additional text.**
- Ensure the output is **valid JSON** without errors or extra characters.
- All keys must be present in the output, even if they are empty.
- **Ignore case sensitivity when matching fields.**
- No incorrect mappings—ensure correctness before assigning values.
- Do not include any markdown syntax or additional text.
 
"""
  