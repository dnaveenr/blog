---
toc: true
layout: post
comments: true
description: Understanding the Drivetrain Approach.
categories: [drivetrain, ML, approach]
title: The Drivetrain Approach
---
## Introduction

The drivetrain approach is introduced by Jeremy Howard along with Margit Zwemer
and Mike Loukidesin in his book “Designing Great Data Products”.

I came across this approach in the FastAI Book and sharing my understanding of the topic here.

## Goal of the Approach

The main goal of the Drivetrain approach is to produce **actionable outcomes** from models. This means 
the results from a model must help and add value to your task.

## Approach

![](http://cdn.oreilly.com/radar/images/posts/0312-1-drivetrain-approach-lg.png)
<center>The four steps in the Drivetrain Approach. Credits: OReilly</center>

The approach is explained as a 4 step process :

1. Have a **clear objective**.
   - What are you trying to achieve ?
2. **Levers** - Actions to be taken to achieve the objective.  
   What is a lever ?  
   By definition, a lever is a handle or bar that is attached to a piece of machinery and which you push or pull in order to operate the machinery.  
   Okay ? But in business terms, it means "initiatives" that are taken to drive the desired impact.  
3. **Data** 
   - What data do we have that we can use ?
   - What data needs to be collected to achieve the objective ?
4. **Models** - Building the models  
   Finally, build a model that we can use to determine the best actions to take to get the best results in terms of the objective.

The first three steps are important and need to be worked on before proceeding to the last step of 
building the models.  
These steps will help you achieve your objective and also lead to generating actionable outcomes from your models which can further be used in the system.

## Example -  Logo Detection System

#### Objective 
   It would be to detect all logos in a give image.
#### Levers
- Get a team of taggers to annotate logos. ( In-house or outsource ?)
- Determine the environment where we will deploying so as to choose the appropriate model
  for latency and compute requirements.
- Is there any pre-trained model we can start off with for this task or do we need to train from scratch ?
  
#### Data
- What all in-house data of logos or images containing logos do we have ?
- Which all publicly available datasets can we use ?
- Do we need to scrap data from the internet for this task ?

#### Model
- Train a logo detection model keeping the levers, data and objective in mind, so we can finally make an actionable outcome.

## Ending

This is small example I have taken for a problem to understand and use the drivetrain approach.
I have presented each step with a series of questions we may have and thereby answering these questions would lead to better understanding of the problem.

If you have any thoughts or questions to ask, comment in the section below.