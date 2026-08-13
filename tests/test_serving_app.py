import io

import pytest

from stages.landcover_model.serving.flask_serving_app import app


@pytest.fixture()
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_root_page_serves_ui(client):
    response = client.get('/')
    assert response.status_code == 200
    html = response.get_data(as_text=True).lower()
    assert 'geovision ai' in html
    assert 'classify' in html


def test_classify_route_requires_file(client):
    response = client.post('/classify', data={})
    assert response.status_code == 400
    body = response.get_json()
    assert 'error' in body
