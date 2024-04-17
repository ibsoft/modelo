import json
import sys


def generate_data(num_records):
    data = []
    for i in range(num_records):
        record = {
            "request": f"What is question {i+1}?",
            "response": f"This is the response to question {i+1}."
        }
        data.append(record)
    return data


def save_to_json(data):
    with open('generated_data.json', 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_json.py <number_of_records>")
        sys.exit(1)

    num_records = int(sys.argv[1])
    data = generate_data(num_records)
    save_to_json(data)
    print(f"Generated {num_records} records and saved to generated_data.json")
