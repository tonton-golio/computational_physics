# Computational Physics

This is a **web app** designed to store notes and resources from the University of Copenhagen's M.Sc. Computational Physics course. We'd appreciate it if people from other speciaizations could chip in too!

[Website link](https://tonton-golio-computational-physics-streamlit-app1home-0d1p3s.streamlitapp.com/)

## Contribution

Please help us fill in information! Be it lecture notes on the subject, or your answers to the questions, we'd love contributions. The app is open source, so feel free to put in a PR, or contact <a href="mailto:aggolles97@gmail.com">Anton</a>.

## Development

The app is built with [Streamlit](https://streamlit.io/), as it's written in python, the language of choice of the program.

After cloning the repository, you need to install the required libraries. We recommended creating a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html) called `venv` in the root directory. Note that the requirements only work with python versions 3.7, 3.8, and 3.9, so make sure you're using one of those in the virtual environment!

Next, you want to move inside the `streamlit_app1` directory with

```bash
cd streamlit_app1
```

Once that's done, install the required libraries with

```bash
pip install -r requirements.txt
```

and start it up with

```python
streamlit run home.py
```

It should automatically start up, and give you a local url. Paste that in your browser, and you're good to go!
