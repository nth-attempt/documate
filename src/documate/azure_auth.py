# src/documate/azure_auth.py

import os
from azure.identity import CertificateCredential

class AzureADTokenManager:
    """
    Manages fetching Azure AD tokens using a certificate credential.
    This class is created once and provides a callable for LangChain.
    """
    def __init__(self):
        """Initializes the CertificateCredential."""
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.certificate_path = os.getenv("AZURE_CERTIFICATE_PATH")

        if not all([self.tenant_id, self.client_id, self.certificate_path]):
            raise ValueError("Azure AD credentials (TENANT_ID, CLIENT_ID, CERTIFICATE_PATH) are not set.")

        if not os.path.exists(self.certificate_path):
            raise FileNotFoundError(f"Azure certificate file not found at: {self.certificate_path}")

        # Read the certificate file content
        with open(self.certificate_path, "rb") as cert_file:
            self.certificate_data = cert_file.read()

        print("Initializing Azure CertificateCredential...")
        self.credential = CertificateCredential(
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            certificate_data=self.certificate_data
        )
        print("CertificateCredential initialized successfully.")

    def get_token(self) -> str:
        """
        Fetches and returns an access token for Azure Cognitive Services.
        This method is the callable passed to LangChain's token provider.
        """
        # The scope for Azure OpenAI
        token = self.credential.get_token("https://cognitiveservices.azure.com/.default")
        return token.token