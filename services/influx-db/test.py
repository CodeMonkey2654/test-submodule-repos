import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.client.query_api import QueryApi

INFLUX_DB_URL = "http://localhost:8086"
INFLUX_DB_TOKEN = "XVQaux-L_QCdgnce7gCiSoxPW6VMt4SscAcBv4btrAar55dnXdRzZhBv6pZnqyFoua5JGqc1-evruGnYTGXaqQ=="
INFLUX_DB_ORG = "Milwaukee School of Engineering"
INFLUX_DB_BUCKET = "RobotTest1"

def test_write():
    client = influxdb_client.InfluxDBClient(url=INFLUX_DB_URL, token=INFLUX_DB_TOKEN, org=INFLUX_DB_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    point = Point("test_measurement").tag("test_tag", "test_value").field("test_field", 42)
    write_api.write(bucket=INFLUX_DB_BUCKET, org=INFLUX_DB_ORG, record=point)
    print("Write test completed.")

def test_read():
    client = influxdb_client.InfluxDBClient(url=INFLUX_DB_URL, token=INFLUX_DB_TOKEN, org=INFLUX_DB_ORG)
    query_api = client.query_api()
    
    query = f'from(bucket:"{INFLUX_DB_BUCKET}") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "test_measurement")'
    result = query_api.query(query=query, org=INFLUX_DB_ORG)
    
    for table in result:
        for record in table.records:
            print(f"Read test result: {record.values}")
    print("Read test completed.")

def test_delete():
    client = influxdb_client.InfluxDBClient(url=INFLUX_DB_URL, token=INFLUX_DB_TOKEN, org=INFLUX_DB_ORG)
    delete_api = client.delete_api()
    
    start = "1970-01-01T00:00:00Z"
    stop = "2100-01-01T00:00:00Z"
    delete_api.delete(start, stop, f'_measurement="test_measurement"', bucket=INFLUX_DB_BUCKET, org=INFLUX_DB_ORG)
    print("Delete test completed.")

def add_dashboard_test_data():
    client = influxdb_client.InfluxDBClient(url=INFLUX_DB_URL, token=INFLUX_DB_TOKEN, org=INFLUX_DB_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    import random
    from datetime import datetime, timedelta

    # Generate data for the last 24 hours
    start_time = datetime.now() - timedelta(hours=24)
    print("Adding dashboard test data. This may take a while...")
    for i in range(1000):  # Write 1000 points
        current_time = start_time + timedelta(minutes=i)
        temperature = random.uniform(20.0, 30.0)
        humidity = random.uniform(40.0, 60.0)
        pressure = random.uniform(990.0, 1010.0)
        
        point = (
            Point("weather_data")
            .tag("location", "Milwaukee")
            .field("temperature", temperature)
            .field("humidity", humidity)
            .field("pressure", pressure)
            .time(current_time)
        )
        write_api.write(bucket=INFLUX_DB_BUCKET, org=INFLUX_DB_ORG, record=point)
    
    print(f"Write test completed. {1000} points written.")

def main():
    print("Starting InfluxDB functionality tests...")
    test_write()
    time.sleep(1)
    test_read()
    test_delete()
    add_dashboard_test_data()
    print("All tests completed.")

if __name__ == "__main__":
    main()