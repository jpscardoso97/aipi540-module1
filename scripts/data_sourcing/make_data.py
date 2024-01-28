# Data sourcing and cleaning
import asyncio
import aiohttp
import aiofiles
import json
import os
import requests
import unicodedata

from bs4 import BeautifulSoup

use_low_res_images = False

image_size = use_low_res_images and "medium" or "original"

curr_dir = os.getcwd()

file_name = 'snakes.txt'
file_path = os.path.join(curr_dir, file_name)

def get_file_path(fname): 
    return os.path.join(curr_dir, fname)

# Get snake species
if not os.path.exists(file_path):

    url = 'https://herpsofnc.org/snakes/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    snake_names = [em.get_text() for em in soup.find_all('em')]

    # fix encodings
    snake_names = [unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8') for name in snake_names]

    # remove newlines
    snake_names = [name.replace('\n', '') for name in snake_names]

    print(snake_names)
    print(len(snake_names))

# Export snake names to text file
try:
    with open(file_path, 'x') as file:
        for name in snake_names:
            file.write(name + '\n')
except FileExistsError:
    print('File already exists.')
except Exception as e:
    print(f'An error occurred: {e}')

# Scrape data from iNaturalist

snake_names_file = 'snakes.txt'
snake_names = []

try:
    with open(get_file_path(snake_names_file), 'r') as file:
        snake_names = file.readlines()
        snake_names = [name.rstrip() for name in snake_names]
        print(snake_names)
except FileNotFoundError:
    print("Error: snakes.txt not found, run nc_snakes notebook")

snake_ids_file = 'snake_ids.json'

api_base_url = "https://api.inaturalist.org/v1"
get_observations_url = api_base_url + "/observations"

def search_species_url(s_name):
    return f"{api_base_url}/search?q={s_name}&sources=taxa"

if os.path.exists(get_file_path(snake_ids_file)):
    print("Snake ids file already exists, skipping...")
    snake_ids = json.load(open(get_file_path(snake_ids_file), 'r'))
else:
    # Get map of snake names to species ids
    snake_ids = {}

    for snake_name in snake_names:
        snake_ids[snake_name] = None

        species_url = search_species_url(snake_name)
        response = requests.get(species_url)
        if response.status_code == 200:
            species = response.json()['results']
            if len(species) > 0:
                snake_ids[snake_name] = species[0]['record']['id']
            else:
                print(f"Error: {snake_name} not found in iNaturalist")
        else:
            print(f"Error {response.status_code} getting {snake_name} from iNaturalist")

            snake_ids

    # save snake_to_species to file
    with open('snake_to_species.json', 'w') as file:
        json.dump(snake_ids, file)


async def download_file(session, folder, url, fname):
    try:
        # Skip if file already exists
        if os.path.exists(os.path.join(folder, fname)):
            print(f"File {fname} already exists, skipping...")
            return
            
        async with session.get(url) as response:
            if response.status == 200:
                fpath = os.path.join(folder, fname)
                async with aiofiles.open(fpath, "wb") as f:
                    await f.write(await response.read())
                    print(f"Downloaded {url} to {fpath}")
            else:
                print(f"Error: {response.status} downloading {url}")
                print(response)
    except aiohttp.ClientResponseError as e:
        print(f"Error: {e.status} downloading {url}")

async def download_all(folder, filenames_and_urls):
    async with aiohttp.ClientSession() as session:
        tasks = [download_file(session, folder, url, filename) for filename, url in filenames_and_urls]
        await asyncio.gather(*tasks)

base_folder_path = os.path.join(curr_dir, "../../data/raw")
if not os.path.exists(base_folder_path):
    raise FileNotFoundError("Base folder path does not exist")

# To avoid downloading images again, set this to False
load_images = True #False

if load_images:
    # Download images for each snake
    for snake_name, snake_id in snake_ids.items():
        if snake_id is None:
            print(f"Skipping {snake_name}, no id found")
            continue
        
        name_without_spaces = snake_name.replace(' ', '_').lower()
        snake_dir = f"{name_without_spaces}-{snake_id}"
        snake_folder_path = os.path.join(base_folder_path, snake_dir)

        if not os.path.exists(snake_folder_path):
            os.mkdir(snake_folder_path)

        params = {
            'verifiable': 'true',
            'taxon_id': snake_id,
            'order_by': 'votes',
            'quality_grade': 'research',
            'per_page': 100
        }
        
        # Get observations for snake
        response = requests.get(get_observations_url, params=params)

        image_urls = {}

        if response.status_code == 200:
            observations = response.json()['results']
            for o in observations:
                photos = o['photos']
                for p in photos:
                    image_url = p['url'].replace('square.', image_size + '.')
                    extension = image_url.split('/')[-1].split('.')[1]
                    filename = f"{p['id']}.{extension}"
                    image_urls[filename] = image_url
        else:
            print(f"Error {response.status_code} getting observations for {snake_name}")

        asyncio.run(download_all(snake_folder_path, image_urls.items())) 
else:
    print("Skipping image download, set load_images to True to download images")