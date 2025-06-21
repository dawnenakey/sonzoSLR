import json

def lambda_handler(event, context):
    """
    Debug Lambda function to see the actual event structure
    """
    # Log the complete event structure
    print("=== DEBUG: Event Structure ===")
    print(f"Event type: {type(event)}")
    print(f"Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
    print(f"Full event: {json.dumps(event, indent=2)}")
    print("=== END DEBUG ===")
    
    # Return a simple response
    return json.dumps({
        'success': True,
        'message': 'Debug function executed',
        'event_keys': list(event.keys()) if isinstance(event, dict) else [],
        'has_httpMethod': 'httpMethod' in event if isinstance(event, dict) else False
    }) 