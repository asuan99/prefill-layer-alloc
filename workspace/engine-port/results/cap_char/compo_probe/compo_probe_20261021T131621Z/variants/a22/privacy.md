| Field | Response |
| :---- | :---- |
| Generatable or reverse engineerable personal data? | No |
| Personal data used to create this model? | No |
| Was consent obtained for any personal data used? | Not Applicable |
| A description of any methods implemented in data acquisition or processing, if any, to address the prevalence of personal data in the training data, where relevant and applicable. | We used only prompts that do not contain any personal data for synthetic data generation. |
| How often is the dataset reviewed? | Before Release |
| Is there provenance for all datasets used in training? | Yes |
| Does data labeling (annotation, metadata) comply with privacy laws? | Yes |
| Is data compliant with data subject requests for data correction or removal, if such a request was made? | No, not possible with externally-sourced data. |
| Applicable Privacy Policy | [NVIDIA Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) |
| During AI model development, strict adherence to copyright policy ensured compliance through risk mitigation and legal reviews. Post-data collection, reserved rights content is identified and removed, with verified opt-out processes for rightsholders. Detailed records document due diligence and transparency. | True |
| We employ automated tools and data processing techniques during pre-training to identify and filter certain categories of personal information. Scans of training datasets detected no PII. | True. We employ automated tools and data processing techniques to scan for Personally Identifiable Information (PII) during pre-training to identify and filter certain categories of personal information, including public-facing contact details such as email addresses and phone numbers. Scans of Common Crawl, CC-News, and Wikimedia datasets did not detect PII in the majority of samples. However, Microsoft Presidio indicated potential findings including business contact information embedded in natural language, such as email addresses and phone numbers. These were removed using verified instances of PII through a combination of automated filtering and human-in-the-loop validation. This evaluation used a 3,000-sample subset per dataset, identified as the optimal threshold for maximizing embedder accuracy.  |