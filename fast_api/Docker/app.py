from flask import Flask, request, render_template_string

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multiplication Table Generator</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f3f4f6;
            --card-bg: #ffffff;
            --text-main: #1f2937;
            --text-muted: #6b7280;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            color: var(--text-main);
        }

        .container {
            background-color: var(--card-bg);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            width: 100%;
            max-width: 400px;
            text-align: center;
        }

        h1 {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 1.5rem;
            color: var(--text-main);
        }

        .input-group {
            margin-bottom: 1.5rem;
        }

        input[type="number"] {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-size: 1rem;
            box-sizing: border-box; /* Ensures padding doesn't affect width */
            outline: none;
            transition: border-color 0.2s;
        }

        input[type="number"]:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }

        button {
            width: 100%;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 0.75rem;
            font-size: 1rem;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.2s;
            font-weight: 600;
        }

        button:hover {
            background-color: var(--primary-hover);
        }

        #resultArea {
            margin-top: 1.5rem;
            text-align: left;
            border-top: 1px solid #e5e7eb;
            padding-top: 1rem;
            max-height: 300px;
            overflow-y: auto;
            {% if not show_results %}display: none;{% endif %}
        }

        .table-row {
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #f3f4f6;
            font-family: 'Courier New', Courier, monospace;
        }

        .table-row:last-child {
            border-bottom: none;
        }

        .table-row:hover {
            background-color: #f9fafb;
        }

        .error {
            color: #ef4444;
            font-size: 0.875rem;
            margin-top: 0.5rem;
            {% if not show_error %}display: none;{% endif %}
        }
    </style>
</head>
<body>

    <div class="container">
        <h1>Multiplication Table</h1>
        
        <form method="POST">
            <div class="input-group">
                <input type="number" id="numberInput" name="number" placeholder="Enter a number (e.g., 5)" value="{{ number if number else '' }}" required>
                <div id="errorMsg" class="error"{% if show_error %} style="display: block;"{% endif %}>Please enter a valid number.</div>
            </div>

            <button type="submit">Generate Table</button>
        </form>

        <div id="resultArea">
            {% if show_results %}
                {% for i in range(1, 11) %}
                    <div class="table-row">
                        <span>{{ number }} x {{ i }}</span>
                        <span>= {{ number * i }}</span>
                    </div>
                {% endfor %}
            {% endif %}
        </div>
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def multiplication_table():
    number = None
    show_results = False
    show_error = False
    
    if request.method == 'POST':
        try:
            number = int(request.form['number'])
            show_results = True
        except (ValueError, TypeError):
            show_error = True
    
    return render_template_string(HTML_TEMPLATE, 
                                 number=number, 
                                 show_results=show_results, 
                                 show_error=show_error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

    