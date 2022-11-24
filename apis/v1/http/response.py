


def respond_500(message):
    print(message, flush=True)
    return {'success': False, message: message}, 500
