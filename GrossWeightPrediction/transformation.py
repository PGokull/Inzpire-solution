import pandas as pd

def transform_data(input_file, output_file):
    df = pd.read_csv(input_file)

    required_columns = ['_id', 'Crate', 'Content', 'NetWeight', 'GrossWeight']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The dataset must contain the '{col}' column.")

    df['GrossWeight'] = df['GrossWeight'].fillna(0)
    df = df[df['NetWeight'] < df['GrossWeight']]

    weight_map = df[['_id', 'Crate', 'NetWeight', 'GrossWeight']].drop_duplicates(subset=['_id'])
    content_counts = df.groupby(['_id','Crate', 'Content']).size().reset_index(name='ContentCount')
    pivot_df = content_counts.pivot(index=['_id','Crate'], columns='Content', values='ContentCount').fillna(0)
    pivot_df = pd.merge(pivot_df, weight_map, on=['_id','Crate'], how='left')
    pivot_df.columns.name = None
    pivot_df.columns = [str(col) for col in pivot_df.columns]
    pivot_df.to_csv(output_file, index=False)
    
def merge_crate_weight(input_file1 , input_file2 , output_file):
    data1 = pd.read_csv(input_file1)
    data2 = pd.read_csv(input_file2)
    combined_data=pd.merge(data1,data2, on='Crate', how='left')
    combined_data.dropna(inplace=True)
    combined_data.drop(columns=['_id','Crate'],axis=1,inplace=True)
    combined_data.to_csv(output_file,index=False)
    print("Done")

def remove_outliers(input_file):
    
    return

input_file = r"D:\Inzpire-Solutions\grossweight\merged_data_original.csv" 
output_file = r"D:\Inzpire-Solutions\grossweight\transformed_data.csv"
#transform_data(input_file, output_file)
merge_crate_weight(
    input_file1=r'D:\Inzpire-Solutions\grossweight\merged_data_original_transformed_outlies.csv',
    input_file2=r'D:\Inzpire-Solutions\grossweight\crate.csv',
    output_file=r'D:\Inzpire-Solutions\grossweight\dataset.csv'
)