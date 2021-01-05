---
toc: true
layout: post
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

The approach is explained as a 4 step process :

1. Have a clear objective.
   - What are you trying to achieve ?
2. Levers - Actions to be taken to achieve the objective.  
   What is a lever ?  
   By definition, a lever is a handle or bar that is attached to a piece of machinery and which you push or pull in order to operate the machinery.  
   Okay ? But in business terms, it means "initiatives" that are taken to drive the desired impact.  
3. Data 
   - What data do we have that we can use ?
   - What data needs to be collected to achieve the objective ?
4. Models - Building the models  
   Finally, build a model that we can use to determine the best actions to take to get the best results in terms of the objective.

The first three steps are important and need to be worked on before proceeding to the last step of 
building the models.  
These steps will help you achieve your objective and also lead to generating actionable outcomes from your models which can further be used in the system.

## Example - Image Search Engine

#### Objective would be return relevant similar images to the query image.
#### Levers
- Good feature representation of the image.
- Have an index of all image features in our database.
- Return images similar to the given query image.
#### Data
- 