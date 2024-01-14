# Описание Датасета
## Информация о датасете

В представленном наборе данных за последние 10 лет содержится информация о ежедневных наблюдениях за погодой в различных местах Австралии.

## Цель и задачи

Прогнозирование дождя на следующий день, обучая классификационные модели целевой переменной RainTomorrow.

**Процесс обработки данных:**
1. **Обработка пропущенных значений:** Первым шагом была проведена проверка наличия пропущенных значений в датасете. Выявленные пропуски были удалены или заполнены, чтобы обеспечить качественную подготовку данных для дальнейшего анализа.

2. **Нормализация данных:** Для обеспечения стабильности и сравнимости различных измерений была применена нормализация данных с использованием метода Min-Max Scaling. Этот шаг позволяет привести значения признаков к одному диапазону и избежать проблемы сильного влияния признаков с большими значениями на модели.

3. **Обработка категориальных признаков:** В данном датасете отсутствуют категориальные признаки, что упрощает процесс обработки данных. В случае наличия категориальных переменных, их требовалось бы обработать с использованием соответствующих методов, таких как кодирование или преобразование.

4. **Удаление коррелирующих признаков:** Для улучшения эффективности модели и избежания мультиколлинеарности были удалены фичи, имеющие высокую корреляцию между собой.
