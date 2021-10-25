import json
import requests
import pandas as pd

if __name__ == '__main__':
    trainer = pd.read_csv('Telecom_Train.csv')

    churn_url = 'http://127.0.0.1:5000'
    churn_dict = dict(trainer.iloc[1])
    churn_dict = {'state': 'OH',
                  'account_length': 1,
                  'area_code': 'area_code_415',
                  'international_plan': 'no',
                  'voice_mail_plan': 'yes',
                  'number_vmail_messages': 26,
                  'total_day_minutes': 161.6,
                  'total_day_calls': 123,
                  'total_day_charge': 27.47,
                  'total_eve_minutes': 195.5,
                  'total_eve_calls': 103,
                  'total_eve_charge': 16.62,
                  'total_night_minutes': 254.4,
                  'total_night_calls': 103,
                  'total_night_charge': 11.45,
                  'total_intl_minutes': 13.7,
                  'total_intl_calls': 3,
                  'total_intl_charge': 500,
                  'number_customer_service_calls': 1,
                  'churn': 'no'}

    # Requesting the API
    churn_json = json.dumps(churn_dict)
    send_requests = requests.post(churn_url, churn_json)
    print(send_requests)
    print(send_requests.json())
