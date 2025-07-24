# NBA Data Pipeline DAG Architecture

## 🏗️ Architecture Overview

This document details the **simplified DAG architecture** that focuses on core data sources while removing salary scraping complexity.

## 📊 DAG Comparison

| Aspect | Monolithic DAG | Split DAGs | Benefit |
|--------|----------------|------------|---------|
| **Failure Isolation** | One failure blocks all | Isolated failures | ✅ Higher reliability |
| **Scheduling** | Single cadence for all | Source-specific cadences | ✅ Optimized resource usage |
| **Maintenance** | All-or-nothing updates | Independent iteration | ✅ Faster development |
| **Monitoring** | Single SLA for everything | Granular SLAs | ✅ Better observability |
| **Parsing Speed** | Large file slows DagBag | Smaller files | ✅ Faster Airflow startup |

## 🗓️ Current DAG Set

| # | DAG file | Purpose | Schedule | SLA | Retries |
|---|----------|---------|----------|-----|---------|
| 1 | `nba_advanced_ingest.py` | Advanced metrics (Basketball‑Reference) | `@daily` | 1 h | 2 |
| 2 | `injury_etl.py`          | Injury CSV processing | `@monthly` | 1 h | 1 |
| 3 | `nba_data_loader.py`     | Load all sources into DuckDB | `@daily` | 3 h | 2 |

> **Salary cap**: the yearly cap/parquet is committed by the build pipeline
> and version‑controlled; no Airflow DAG is required.

### Dependency graph

```
nba_advanced_ingest ┐
injury_etl ├──► nba_data_loader
```

## 🗓️ DAG Scheduling Strategy

### 1. `nba_advanced_ingest` - Daily
**Rationale**: Advanced stats update daily
- **Schedule**: `@daily`
- **SLA**: 1 hour
- **Retries**: 2 with 5-minute delays
- **Sources**: Basketball-Reference

### 2. `injury_etl` - Monthly
**Rationale**: Injury data updates monthly
- **Schedule**: `@monthly`
- **SLA**: 1 hour
- **Retries**: 1 with 5-minute delays
- **Sources**: Local CSV files

### 3. `nba_data_loader` - Daily
**Rationale**: Loads all data into DuckDB daily
- **Schedule**: `@daily`
- **SLA**: 3 hours
- **Dependencies**: Advanced metrics and injury ETL via ExternalTaskSensor

## 🔗 Dependency Management

### ExternalTaskSensor Configuration

```python
# Wait for advanced metrics
wait_advanced = ExternalTaskSensor(
    task_id="wait_advanced_ingest",
    external_dag_id="nba_advanced_ingest",
    external_task_id="scrape_advanced_metrics",
    timeout=3600,                     # 1 hour timeout
    mode="reschedule",
    poke_interval=300,                # Check every 5 minutes
)
```

### Timeout Strategy

| DAG | Timeout | Rationale |
|-----|---------|-----------|
| Daily DAGs | 1 hour | Normal operation time |
| Monthly DAGs | 2 hours | Allow for monthly task completion |

## 📈 Performance Metrics

### Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Ingest Success Rate** | >95% | Successful DAG runs / Total runs |
| **Data Quality** | >99% | Valid rows / Total rows |
| **SLA Compliance** | >90% | On-time completions / Total runs |

### Monitoring Dashboard

```sql
-- DAG Performance Query
SELECT 
    dag_id,
    COUNT(*) as total_runs,
    AVG(CASE WHEN state = 'success' THEN 1 ELSE 0 END) as success_rate,
    AVG(duration) as avg_duration_minutes
FROM airflow.task_instance 
WHERE start_date >= CURRENT_DATE - 30
GROUP BY dag_id;
```

## 🔄 Removed Components

### Salary Scraping (Removed)
- ❌ `nba_salary_ingest.py` - Player & team salary scraping
- ❌ `salary_cap_snapshot.py` - Yearly salary cap scraping
- ❌ ESPN/HoopsHype scrapers in `scrape_utils.py`

### Salary Cap Handling (Updated)
- ✅ **Build pipeline**: Yearly cap data committed to version control
- ✅ **No DAG required**: Parquet files pre-baked by build process
- ✅ **Loader compatibility**: Still loads cap data if available

## 🛠️ Implementation Details

### Error Handling Strategy

1. **Primary Source Failure**: Graceful degradation when data unavailable
2. **Rate Limiting**: Exponential backoff with jitter
3. **Data Validation**: Quality gates before loading to DuckDB
4. **Alerting**: Email notifications for critical failures

### Retry Configuration

```python
default_args = dict(
    retries=2,                           # Standard retries
    retry_delay=timedelta(minutes=5),    # Standard delays
    sla=timedelta(hours=1),              # Standard SLA
)
```

### Data Quality Gates

```python
# Quality checks before loading
if len(df) == 0:
    raise ValueError(f"No data found for season {season}")

required_cols = ["Season", "Player", "Team"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")
```

## 📊 Cost-Benefit Analysis

### Pros of Simplified Architecture

| Benefit | Impact | Metric |
|---------|--------|--------|
| **Reliability** | High | 95%+ uptime per source |
| **Maintainability** | High | Independent development cycles |
| **Simplicity** | High | Fewer DAGs to manage |
| **Monitoring** | High | Granular observability |

### Cons of Simplified Architecture

| Drawback | Mitigation | Status |
|----------|------------|--------|
| **Less data sources** | External salary data | ✅ Addressed |
| **Reduced functionality** | Core metrics preserved | ✅ Minimized |

## 🚀 Deployment Checklist

### Pre-Deployment
- [x] All DAG files created and tested
- [x] Salary scraping removed and stubbed
- [x] ExternalTaskSensor dependencies configured
- [x] Data quality gates implemented
- [x] Monitoring and alerting configured

### Deployment
- [x] Deploy new DAGs to Airflow
- [x] Disable old monolithic DAG
- [x] Verify all DAGs are running
- [x] Check data flow end-to-end
- [x] Monitor for 24 hours

### Post-Deployment
- [x] Compare performance metrics
- [x] Validate data quality
- [x] Update documentation
- [x] Train team on new architecture

## 📚 References

- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [ExternalTaskSensor Guide](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/sensors.html)
- [DAG Design Patterns](https://medium.com/@gharikrishnade/airflow-dag-design-patterns-keeping-it-clean-and-modular-ae07bf9b6f11) 