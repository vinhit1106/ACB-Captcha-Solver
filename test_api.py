import unittest

import index


class TransactionApiTests(unittest.TestCase):
    def test_parser_keeps_transaction_and_description_together(self):
        html = """
        <table id="table1">
          <tr class="table-style-double1">
            <td>01/08/2026</td><td>01/08/2026</td><td>TX-1</td>
            <td>1.000</td><td></td><td>20.000</td>
          </tr>
          <tr class="table-style-double1"><td class="acctSum">Test transfer</td></tr>
        </table>
        """
        self.assertEqual(index.ACBClient.parse_transactions(html), [{
            "effectiveDate": "01/08/2026", "transactionDate": "01/08/2026",
            "transactionNumber": "TX-1", "debit": 1000, "credit": 0,
            "balance": 20000, "description": "Test transfer",
        }])

    def test_endpoint_rejects_missing_credentials_without_contacting_acb(self):
        response = index.app.test_client().post("/api/transactions", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"], "credentials_required")

    def test_trusted_cookie_is_loaded_into_the_acb_domain_only(self):
        client = index.ACBClient(None, None, 2, cookie_header="token=abc; JSESSIONID=session", session_id="dse")
        cookies = client.http.cookies.get_dict(domain="online.acb.com.vn")
        self.assertEqual(cookies, {"token": "abc", "JSESSIONID": "session"})
        self.assertEqual(client.session_id, "dse")

    def test_endpoint_requires_a_complete_trusted_session(self):
        response = index.app.test_client().post("/api/transactions", json={"cookie": "token=abc"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"], "trusted_session_incomplete")


if __name__ == "__main__":
    unittest.main()
