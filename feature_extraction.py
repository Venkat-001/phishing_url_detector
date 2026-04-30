# feature_extraction.py
import numpy as np
import re
from urllib.parse import urlparse
import tldextract

def extract_features_from_url(url, row=None):
    try:
        parsed = urlparse(url)
        ext = tldextract.extract(url)

        url_length = len(url)
        num_digits = sum(c.isdigit() for c in url)
        num_special = sum(not c.isalnum() for c in url)
        num_subdomains = len(ext.subdomain.split('.')) if ext.subdomain else 0
        domain_length = len(ext.domain)
        tld_length = len(ext.suffix) if ext.suffix else 0
        is_https = 1 if url.lower().startswith("https") else 0
        num_dots = url.count('.')
        num_hyphens = url.count('-')
        num_slashes = url.count('/')
        has_at = 1 if '@' in url else 0
        has_ip = 1 if re.match(r'http[s]?://\d+\.\d+\.\d+\.\d+', url) else 0
        path_length = len(parsed.path) if parsed.path else 0
        query_length = len(parsed.query) if parsed.query else 0

        # FIXED: Only keeping very specific phishing-only phrases.
        # Removed broad words like login, secure, account, signin, verify,
        # free, lucky, bonus, click — these appear in legitimate URLs too
        # and were the main cause of everything being flagged as phishing.
        suspicious_keywords = [
            'ebayisapi',
            'webscr',
            'cmd=_login',
            'verify-account',
            'update-billing',
            'confirm-payment',
            'banking-login',
            'password-reset',
            'secure-login',
            'account-verify',
        ]
        has_suspicious = 1 if any(kw in url.lower() for kw in suspicious_keywords) else 0

        return np.array([
            url_length,
            num_digits,
            num_special,
            num_subdomains,
            domain_length,
            tld_length,
            is_https,
            num_dots,
            num_hyphens,
            num_slashes,
            has_at,
            has_ip,
            path_length,
            query_length,
            has_suspicious
        ], dtype=float)

    except Exception:
        return np.zeros(15, dtype=float)


def extract_features_from_dataframe(df):
    df.columns = df.columns.str.strip().str.lower()
    label_cols = [col for col in df.columns if "label" in col]
    if not label_cols:
        raise ValueError("No label column found.")
    label_column = label_cols[0]
    print(f"Using '{label_column}' as label column")
    if 'url' not in df.columns:
        raise ValueError("Dataset must have a 'url' column.")
    X_list = []
    for i, url in enumerate(df['url']):
        feats = extract_features_from_url(str(url))
        X_list.append(feats)
        if i % 50000 == 0:
            print(f"Processed {i}/{len(df)} URLs...")
    X = np.array(X_list)
    y_raw = df[label_column].astype(str).str.lower().str.strip()
    y = y_raw.map({"good": 0, "bad": 1}).values
    print(f"Feature extraction complete. Shape: {X.shape}")
    return X, y


def extract_features_for_prediction(url):
    return extract_features_from_url(str(url)).reshape(1, -1)