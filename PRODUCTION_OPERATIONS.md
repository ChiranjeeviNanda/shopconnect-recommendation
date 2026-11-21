# Production Operations

## Batch and On-Request Scoring

To move this project into production, a hybrid setup combining batch and real-time scoring makes the most sense:

* **Daily batch scoring:** An AWS Batch job runs once a day to pre-compute top recipes for active households and saves them to DynamoDB for fast lookups.

* **On-request live scoring:** API Gateway sends requests to AWS Lambda when a household is new, changes dislikes, swaps a meal, or has no cached data. The service reads household data and recent events from Aurora PostgreSQL. It then runs the existing filters and ranking rules.


## Keeping Data Fresh

* **Scheduled updates:** EventBridge triggers AWS Glue or AWS Batch jobs to ingest new recipe data and refresh caches on a regular schedule.

* **Event-driven updates:** User interactions and profile edits stream through Kinesis, triggering Lambda functions to update popularity scores and recent history.


## Core Infrastructure & Fallbacks

* **Storage & Secrets:** **S3** holds raw datasets and pipeline outputs, **Aurora/RDS** stores the main PostgreSQL database, **DynamoDB/Redis** serves cached recommendations, and **Secrets Manager** protects database credentials.

* **Monitoring & AI Resilience:** **CloudWatch** tracks pipeline logs and alerts. **OpenAI/Claude/Gemini** runs only after recipes are ranked to generate explanations, and the app falls back to rule-based text if the API is down.