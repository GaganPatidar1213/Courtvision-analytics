"""Shared presentation components for the COURTSIDE / LAB workstation."""

from html import escape
from pathlib import Path

import streamlit as st

PAGES = (
    ("app.py", "Home", ""),
    ("pages/1_Dataset_Explorer.py", "Dataset Explorer", "Dataset_Explorer"),
    ("pages/2_Preprocessing.py", "Data Preprocessing", "Preprocessing"),
    ("pages/3_Descriptive_Analytics.py", "Descriptive Analytics", "Descriptive_Analytics"),
    ("pages/4_Visualizations.py", "Data Visualizations", "Visualizations"),
    ("pages/5_Apriori.py", "Apriori", "Apriori"),
    ("pages/6_Decision_Tree.py", "Decision Tree", "Decision_Tree"),
    ("pages/7_A_CLOSE.py", "A-CLOSE", "A_CLOSE"),
    ("pages/8_Project_Report.py", "Project Report", "Project_Report"),
)
SOURCE = "https://archive.ics.uci.edu/dataset/587/basketball+dataset"
STYLE = Path(__file__).with_name("courtside.css").read_text(encoding="utf-8-sig")


def chrome(active):
    st.html(f"<style>{STYLE}</style>")
    with st.sidebar:
        st.html('<div class="lab-brand">COURTSIDE<span> / </span>LAB<small>BASKETBALL RESEARCH</small></div>')
        for index, (path, name, _) in enumerate(PAGES):
            marker = " ›" if name == active else ""
            help_text = "Current page" if name == active else f"Open {name}"
            st.page_link(path, label=f"{index:02d}  {name}{marker}", help=help_text)
        st.html('<div class="lab-sidebar-footer">SOURCE ARCHIVE</div>')
        st.markdown(f"[UCI Basketball Dataset · 587]({SOURCE})")
    number = next(index for index, (_, name, _) in enumerate(PAGES) if name == active)
    if number:
        micro(f"{number:02d} / {active}")


def micro(label):
    st.html(f'<div class="lab-micro">{escape(label)}</div>')


def section(number, title, anchor=None):
    st.html(f'<div class="lab-rule"><span>{escape(number)}</span></div>')
    st.header(title, anchor=anchor)


def facts(entries):
    items = "".join(
        f'<div><dt>{escape(str(label))}</dt><dd>{escape(str(value))}</dd></div>'
        for label, value in entries
    )
    st.html(f'<dl class="lab-facts">{items}</dl>')


def annotations(findings, heading="Research annotations"):
    notes = "".join(
        f'<li><span aria-hidden="true">{index:02d}</span><p>{escape(finding)}</p></li>'
        for index, finding in enumerate(findings, 1)
    )
    st.html(f'<aside class="lab-notes" aria-label="{escape(heading)}"><h3>{escape(heading)}</h3>'
            f'<ol>{notes}</ol></aside>')


def relationship(left_label, left, right_label, right, detail):
    st.html(f"""
        <div class="lab-relation">
          <div><span>{escape(left_label)}</span><strong>{escape(left)}</strong></div>
          <div class="lab-arrow" aria-hidden="true">→</div>
          <div><span>{escape(right_label)}</span><strong>{escape(right)}</strong></div>
          <p>{escape(detail)}</p>
        </div>
    """)


def court():
    positions = [(18, 77), (17, 49), (32, 24), (58, 20), (81, 45), (81, 76), (61, 57), (51, 82)]
    stations = ["Archive", "Preparation", "Field notes", "Gallery", "Apriori", "Decision tree", "A-CLOSE", "Report"]
    marks, key = [], []
    for index, ((x, y), short, (_, name, route)) in enumerate(zip(positions, stations, PAGES[1:]), 1):
        marks.append(f'<a class="lab-marker" style="--x:{x}%;--y:{y}%" href="{route}" '
                     f'target="_self" aria-label="{index:02d} {name}" title="{name}">{index:02d}</a>')
        key.append(f'<a href="{route}" target="_self"><b>{index:02d}</b> {short}</a>')
    st.html(f"""
      <section class="lab-court" aria-label="Half-court navigation">
        <div class="lab-court-head"><span>COURT MAP</span><span>HALF COURT / 08 STATIONS</span></div>
        <div class="lab-court-map">
          <svg viewBox="0 0 600 390" aria-hidden="true" focusable="false">
            <defs>
              <pattern id="lab-floor" width="24" height="24" patternUnits="userSpaceOnUse">
                <path d="M24 0H0V24" fill="none" stroke="#39302b" stroke-width="1"/>
              </pattern>
            </defs>
            <rect x="16" y="16" width="568" height="358" rx="2" fill="url(#lab-floor)" stroke="#bc825e"/>
            <path d="M217 374V255H383V374Z" fill="#F07837" fill-opacity=".10" stroke="#bc825e"/>
            <path d="M68 374V332A233 233 0 0 1 532 332V374M231 255a69 69 0 0 1 138 0M276 348h48"
                  fill="none" stroke="#bc825e" stroke-width="1.5"/>
            <path d="M231 255a69 69 0 0 0 138 0" fill="none" stroke="#bc825e" stroke-dasharray="5 6"/>
            <path d="M221 16a79 79 0 0 0 158 0" fill="none" stroke="#bc825e"/>
            <path d="M261 348a39 39 0 0 1 78 0" fill="none" stroke="#bc825e"/>
            <circle cx="300" cy="337" r="7" fill="none" stroke="#f07837" stroke-width="2"/>
          </svg>
          {"".join(marks)}
        </div>
        <nav class="lab-court-key" aria-label="Court station index">{"".join(key)}</nav>
      </section>
    """)
