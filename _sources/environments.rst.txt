Environments
=====================================

There are 65 environments in the dataset, covering a wide range of scenarios, including urban, rural, Domes, infrastructure, Them, and nature. Each environment is designed to provide a unique setting for training and testing autonomous systems. The environments vary in size, complexity, and type of content. Some environments are split into multiple sub-environments to provide more diverse scenarios, such as weather variations or different times of day.

You can find a general overview of the environments in the following table. Each environment is listed with its name, number of trajectories, type (Infra - Infrastructure, Domes - Domestic, Rural, Them - Thematic, Nature, Urban), whether it is indoor or outdoor, size, intersting features, and a preview image.

.. list-table::
   :widths: 10 5 15 10 5 15 40
   :header-rows: 1

   * - Environment
     - Traj
     - Type
     - In/Out
     - Size
     - Feature
     - Preview
   * - | Abandoned
       | Cable
     - 8
     - Infra
     - Mix
     - Large
     - Weather
     - .. image:: images/env_preview/AbandonedCable.gif
   * - | Abandoned
       | Factory
     - 9
     - Infra
     - Outdoor
     - Medium
     - Dusty
     - .. image:: images/env_preview/AbandonedFactory.gif
   * - | Abandoned
       | Factory2
     - 6
     - Infra
     - Indoor
     - Small
     - 
     - .. image:: images/env_preview/AbandonedFactory2.gif
   * - | Abandoned
       | School
     - 10
     - Infra
     - Mix
     - Large
     - 
     - .. image:: images/env_preview/AbandonedSchool.gif
   * - | American
       | Diner
     - 5
     - Domes
     - Indoor
     - Medium
     - 
     - .. image:: images/env_preview/AmericanDiner.gif
   * - | Amusement
       | Park
     - 8
     - Rural
     - Outdoor
     - Medium
     - 
     - .. image:: images/env_preview/AmusementPark.gif
   * - | Ancient
       | Towns
     - 7
     - Rural
     - Outdoor
     - Medium
     - 
     - .. image:: images/env_preview/AncientTowns.gif
   * - Antiquity3D
     - 10
     - Them
     - Outdoor
     - Large
     - Day/Night
     - .. image:: images/env_preview/Antiquity3D.gif
   * - Apocalyptic
     - 5
     - Them
     - Outdoor
     - Medium
     - 
     - .. image:: images/env_preview/Apocalyptic.gif
   * - | ArchVizTiny
       | HouseDay
     - 7
     - Domes
     - Indoor
     - Small
     - Day/Night
     - .. image:: images/env_preview/ArchVizTinyHouseDay.gif
   * - | ArchVizTiny
       | HouseNight
     - 7
     - Domes
     - Indoor
     - Small
     - Day/Night
     - .. image:: images/env_preview/ArchVizTinyHouseNight.gif
   * - | Brushify
       | Moon
     - 6
     - Nature
     - Outdoor
     - Large
     - 
     - .. image:: images/env_preview/BrushifyMoon.gif
   * - CarWelding
     - 9
     - Infra
     - Indoor
     - Medium
     - Dynamic
     - .. image:: images/env_preview/CarWelding.gif
   * - | Castle
       | Fortress
     - 12
     - Rural
     - Mix
     - Large
     - 
     - .. image:: images/env_preview/CastleFortress.gif
   * - CoalMine
     - 6
     - Infra
     - Indoor
     - Medium
     - 
     - .. image:: images/env_preview/CoalMine.gif
   * - | Construction
       | Site
     - 10
     - Infra
     - Outdoor
     - Medium
     - Lighting
     - .. image:: images/env_preview/ConstructionSite.gif
   * - | Country
       | House
     - 6
     - Domes
     - Indoor
     - Small
     - 
     - .. image:: images/env_preview/CountryHouse.gif
   * - Cyberpunk
     - 8
     - Them
     - Outdoor
     - Medium
     - 
     - .. image:: images/env_preview/CyberPunkDowntown.gif
   * - | CyberPunk
       | Downtown
     - 7
     - Them
     - Indoor
     - Medium
     - Dynamic
     - .. image:: images/env_preview/Cyberpunk.gif
   * - | Desert
       | GasStation
     - 5
     - Rural
     - Outdoor
     - Small
     - Lighting
     - .. image:: images/env_preview/DesertGasStation.gif
   * - Downtown
     - 9
     - Urban
     - Outdoor
     - Medium
     - 
     - .. image:: images/env_preview/Downtown.gif
   * - | EndofThe
       | World
     - 6
     - Rural
     - Outdoor
     - Medium
     - Dusty
     - .. image:: images/env_preview/EndofTheWorld.gif
   * - | Factory
       | Weather
     - 14
     - Infra
     - Outdoor
     - Large
     - Weather
     - .. image:: images/env_preview/FactoryWeather.gif
   * - Fantasy
     - 7
     - Rural
     - Outdoor
     - Medium
     - 
     - .. image:: images/env_preview/Fantasy.gif
   * - ForestEnv
     - 9
     - Nature
     - Outdoor
     - Large
     - 
     - .. image:: images/env_preview/ForestEnv.gif
   * - Gascola
     - 10
     - Nature
     - Outdoor
     - Medium
     - Day/Night
     - .. image:: images/env_preview/Gascola.gif
   * - GothicIsland
     - 11
     - Rural
     - Mix
     - Large
     - 
     - .. image:: images/env_preview/GothicIsland.gif
   * - GreatMarsh
     - 10
     - Rural
     - Outdoor
     - Medium
     - Fog
     - .. image:: images/env_preview/GreatMarsh.gif
   * - HongKong
     - 5
     - Urban
     - Outdoor
     - Small
     - Night
     - .. image:: images/env_preview/HongKong.gif
   * - Hospital
     - 11
     - Infra
     - Indoor
     - Large
     - 
     - .. image:: images/env_preview/Hospital.gif
   * - House
     - 8
     - Domes
     - Indoor
     - Medium
     - 
     - .. image:: images/env_preview/House.gif
   * - | HQWestern
       | Saloon
     - 8
     - Rural
     - Mix
     - Medium
     - 
     - .. image:: images/env_preview/HQWesternSaloon.gif
   * - | Industrial
       | Hangar
     - 7
     - Infra
     - Outdoor
     - Medium
     - 
     - .. image:: images/env_preview/IndustrialHangar.gif
   * - | Japanese
       | Alley
     - 7
     - Urban
     - Mix
     - Small
     - | Rain
       | Night
     - .. image:: images/env_preview/JapaneseAlley.gif
   * - JapaneseCity
     - 11
     - Urban
     - Mix
     - Medium
     - 
     - .. image:: images/env_preview/JapaneseCity.gif
   * - MiddleEast
     - 12
     - Urban
     - Outdoor
     - Medium
     - Weather
     - .. image:: images/env_preview/MiddleEast.gif
   * - | ModernCity
       | Downtown
     - 9
     - Urban
     - Mix
     - Medium
     - 
     - .. image:: images/env_preview/ModernCityDowntown.gif
   * - | Modular
       | Neighborhood
     - 11
     - Urban
     - Outdoor
     - Large
     - 
     - .. image:: images/env_preview/ModularNeighborhood.gif
   * - | Modular
       | Neighborhood
       | IntExt
     - 8
     - Urban
     - Mix
     - Large
     - 
     - .. image:: images/env_preview/ModularNeighborhoodIntExt.gif
   * - | ModUrban
       | City
     - 12
     - Urban
     - Mix
     - Medium
     - 
     - .. image:: images/env_preview/ModUrbanCity.gif
   * - | Nordic
       | Harbor
     - 8
     - Urban
     - Outdoor
     - Medium
     - 
     - .. image:: images/env_preview/NordicHarbor.gif
   * - Ocean
     - 9
     - Nature
     - Outdoor
     - Medium
     - Dynamic
     - .. image:: images/env_preview/Ocean.gif
   * - Office
     - 7
     - Domes
     - Indoor
     - Medium
     - 
     - .. image:: images/env_preview/Office.gif
   * - | OldBrick
       | HouseDay
     - 7
     - Domes
     - Indoor
     - Small
     - Day/Night
     - .. image:: images/env_preview/OldBrickHouseDay.gif
   * - | OldBrick
       | HouseNight
     - 6
     - Domes
     - Indoor
     - Small
     - Day/Night
     - .. image:: images/env_preview/OldBrickHouseNight.gif
   * - | OldIndustrial
       | City
     - 10
     - Infra
     - Outdoor
     - Medium
     - 
     - .. image:: images/env_preview/OldIndustrialCity.gif
   * - | Old
       | Scandinavia
     - 8
     - Nature
     - Outdoor
     - Large
     - Day/Night
     - .. image:: images/env_preview/OldScandinavia.gif
   * - | OldTown
       | Fall
     - 3
     - Urban
     - Outdoor
     - Medium
     - Season
     - .. image:: images/env_preview/OldTownFall.gif
   * - | OldTown
       | Night
     - 3
     - Urban
     - Outdoor
     - Medium
     - Season
     - .. image:: images/env_preview/OldTownNight.gif
   * - | OldTown
       | Summer
     - 3
     - Urban
     - Outdoor
     - Medium
     - Season
     - .. image:: images/env_preview/OldTownSummer.gif
   * - | OldTown
       | Winter
     - 3
     - Urban
     - Outdoor
     - Medium
     - Season
     - .. image:: images/env_preview/OldTownWinter.gif
   * - PolarSciFi
     - 9
     - Them
     - Mix
     - Medium
     - Snow
     - .. image:: images/env_preview/PolarSciFi.gif
   * - Prison
     - 12
     - Infra
     - Mix
     - Large
     - 
     - .. image:: images/env_preview/Prison.gif
   * - Restaurant
     - 9
     - Domes
     - Indoor
     - Medium
     - 
     - .. image:: images/env_preview/Restaurant.gif
   * - RetroOffice
     - 6
     - Domes
     - Indoor
     - Small
     - 
     - .. image:: images/env_preview/RetroOffice.gif
   * - Rome
     - 7
     - Them
     - Outdoor
     - Medium
     - 
     - .. image:: images/env_preview/Rome.gif
   * - Ruins
     - 9
     - Nature
     - Outdoor
     - Medium
     - 
     - .. image:: images/env_preview/Ruins.gif
   * - SeasideTown
     - 8
     - Rural
     - Outdoor
     - Medium
     - 
     - .. image:: images/env_preview/SeasideTown.gif
   * - | Seasonal
       | Forest
       | Autumn
     - 3
     - Nature
     - Outdoor
     - Large
     - Season
     - .. image:: images/env_preview/SeasonalForestAutumn.gif
   * - | Seasonal
       | Forest
       | Spring
     - 4
     - Nature
     - Outdoor
     - Large
     - Season
     - .. image:: images/env_preview/SeasonalForestSpring.gif
   * - | Seasonal
       | Forest
       | SummerNight
     - 2
     - Nature
     - Outdoor
     - Large
     - | Season
       | Night
     - .. image:: images/env_preview/SeasonalForestSummerNight.gif
   * - | Seasonal
       | Forest
       | Winter
     - 3
     - Nature
     - Outdoor
     - Large
     - Season
     - .. image:: images/env_preview/SeasonalForestWinter.gif
   * - | Seasonal
       | Forest
       | WinterNight
     - 3
     - Nature
     - Outdoor
     - Large
     - | Season
       | Night
     - .. image:: images/env_preview/SeasonalForestWinterNight.gif
   * - Sewerage
     - 12
     - Infra
     - Indoor
     - Large
     - 
     - .. image:: images/env_preview/Sewerage.gif
   * - ShoreCaves
     - 8
     - Nature
     - Outdoor
     - Small
     - 
     - .. image:: images/env_preview/ShoreCaves.gif
   * - Slaughter
     - 12
     - Them
     - Mix
     - Medium
     - 
     - .. image:: images/env_preview/Slaughter.gif
   * - SoulCity
     - 9
     - Rural
     - Mix
     - Medium
     - Dynamic
     - .. image:: images/env_preview/SoulCity.gif
   * - Supermarket
     - 8
     - Domes
     - Indoor
     - Medium
     - Rain
     - .. image:: images/env_preview/Supermarket.gif
   * - | Terrain
       | Blending
     - 4
     - Nature
     - Outdoor
     - Small
     - 
     - .. image:: images/env_preview/TerrainBlending.gif
   * - | Urban
       | Construction
     - 5
     - Urban
     - Outdoor
     - Medium
     - 
     - .. image:: images/env_preview/UrbanConstruction.gif
   * - | Victorian
       | Street
     - 8
     - Urban
     - Outdoor
     - Medium
     - 
     - .. image:: images/env_preview/VictorianStreet.gif
   * - | WaterMill
       | Day
     - 6
     - Rural
     - Outdoor
     - Medium
     - Day/Night
     - .. image:: images/env_preview/WaterMillDay.gif
   * - | WaterMill
       | Night
     - 7
     - Rural
     - Outdoor
     - Medium
     - Day/Night
     - .. image:: images/env_preview/WaterMillNight.gif
   * - | Western
       | Desert
       | Town
     - 12
     - Rural
     - Mix
     - Large
     - Day/Night
     - .. image:: images/env_preview/WesternDesertTown.gif











































































